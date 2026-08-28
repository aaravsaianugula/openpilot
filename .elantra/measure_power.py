#!/usr/bin/env python3
"""
What does the card on the chestnut actually pull? Measure it on the bench, before the car.

Port stage 8 (`.elantra/HANDOFF-egpu-port.md`). The RX 6600 XT is a 160 W TBP part on a single
8-pin, so this is expected to be comfortable -- which is exactly why it gets measured instead of
assumed. Nothing here decides anything on its own: it produces four numbered phases of real
samples plus the CSV behind them, and the go/no-go is read off those.

The dock carries an INA231 behind the ASM2464 bridge. openpilot already reads it in two places
and this uses the same protocol they do, not a re-derivation:

    openpilot/selfdrive/modeld/modeld.py:122          struct.unpack('<Hh', bytes(asm.usb.control_read(0xC0, 5))[:4])
    openpilot/sunnypilot/egpu/chestnut_state.py:38    ASM_SUPPLY_REQUEST = 0xC0

Vendor IN request 0xC0, wLength 5, decoding `<H h B` = bus millivolts, shunt milliamps, fault.
Those two call sites drop the fifth byte because cereal's chestnutState has nowhere to put it.
Here the fault byte -- `bob_flt` in `.elantra/EGPU.md` -- is the whole point: any non-zero value
stops the run. It is a converter fault, and averaging over one is how a bad supply reaches a car.

Why this opens the bridge the way it does
-----------------------------------------
tinygrad's `USB3.__init__` detaches the kernel driver, **resets the device**, sets the
configuration and claims interface 0 -- correct for the process that is going to drive the GPU,
and wrong for a meter. Two of the four phases are measured while something else owns the bridge:
`model` needs modeld running, `gemm` needs a workload looping. A second claim would fail, and the
reset inside it would yank the device out from under a live AM driver.

So this opens the device handle and issues the one vendor IN transfer on endpoint 0. No claim, no
detach, no reset, no configuration change -- nothing that alters the device another process is
driving. It is still extra ep0 traffic alongside that driver's own control transfers, which is
why this is a bench tool: run it offroad, with the car not driving.

This tool does not start the workload. The phase name is a label you assert; it selects a default
duration and prints the preconditions you are claiming to have met.

Usage (on the comma four, dock attached, offroad):
    python .elantra/measure_power.py idle
    python .elantra/measure_power.py model --duration 600
    python .elantra/measure_power.py gemm
    python .elantra/measure_power.py coldstart          # start this BEFORE bringing the card up
    python .elantra/measure_power.py --self-test        # arithmetic only -- measures nothing

Exit codes: 0 clean, 1 fault byte set, 2 nothing could be measured, 3 measured but degraded
(interrupted, or the achieved sample rate fell short of the requested one).
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import statistics
import struct
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Vendor IN on the ASM2464 carrying the INA231. The request number and the USB request-type byte
# are both 0xC0 by coincidence: 0xC0 as request_type is IN | vendor | device, which is what
# tinygrad's USB3.control_read() sends, and 0xC0 as bRequest is the supply-telemetry request.
ASM_SUPPLY_REQUEST = 0xC0
ASM_SUPPLY_LEN = 5
ASM_SUPPLY_FMT = '<HhB'
USB_CTRL_IN_VENDOR_DEVICE = 0xC0

DEFAULT_RATE_HZ = 100.0            # .elantra/EGPU.md stage 3 asks for >= 100 Hz off the INA231
DEFAULT_USB_TIMEOUT_MS = 200
DEFAULT_RECONNECT_TIMEOUT_S = 5.0
# Below this fraction of the requested rate the run still reports its real samples, but exits
# non-zero: a peak found at 4 Hz must never be published as a 100 Hz peak.
RATE_SHORTFALL_FRACTION = 0.9

EXIT_OK = 0
EXIT_FAULT = 1
EXIT_NO_MEASUREMENT = 2
EXIT_DEGRADED = 3


class SupplyUnavailable(Exception):
  """The bridge cannot be reached at all, so no measurement is possible."""


class SupplyReadError(Exception):
  """One read failed. Recoverable: the sampler records a gap and keeps going."""


@dataclass(frozen=True)
class Phase:
  name: str
  default_duration_s: float
  precondition: str


PHASES: dict[str, Phase] = {
  "idle": Phase("idle", 60.0,
                "card powered and enumerated, nothing driving it -- modeld stopped, no tinygrad process holding the bridge"),
  "model": Phase("model", 600.0,
                 "the driving model running on the eGPU at 20 Hz. Ten minutes of this is the number that decides the purchase"
                 + " (.elantra/EGPU.md stage 3)"),
  "gemm": Phase("gemm", 120.0,
                "a saturating FP16 GEMM looping on the card, hot and steady before you start this"),
  "coldstart": Phase("coldstart", 30.0,
                     "START THIS FIRST, then bring the card up. The inrush is at the front of the window,"
                     + " so the sampler has to already be running"),
}


@dataclass(frozen=True)
class Sample:
  index: int
  t: float
  millivolts: int
  milliamps: int
  fault: int
  raw: bytes

  @property
  def volts(self) -> float:
    return self.millivolts / 1000.0

  @property
  def amps(self) -> float:
    return self.milliamps / 1000.0

  @property
  def watts(self) -> float:
    return self.millivolts * self.milliamps / 1e6


@dataclass(frozen=True)
class Gap:
  index: int
  t: float
  detail: str


@dataclass
class PhaseRun:
  phase: str
  requested_rate_hz: float
  requested_duration_s: float
  samples: list[Sample]
  gaps: list[Gap]
  fault: Sample | None
  interrupted: bool
  link_lost: bool
  started_utc: str = ""

  @property
  def elapsed_s(self) -> float:
    return self.samples[-1].t - self.samples[0].t if len(self.samples) >= 2 else 0.0

  @property
  def achieved_rate_hz(self) -> float | None:
    """Intervals per second actually achieved, or None when there is nothing to divide by."""
    return (len(self.samples) - 1) / self.elapsed_s if len(self.samples) >= 2 and self.elapsed_s > 0 else None

  @property
  def max_interval_s(self) -> float:
    ts = [s.t for s in self.samples]
    return max((b - a for a, b in zip(ts, ts[1:], strict=False)), default=0.0)


@dataclass(frozen=True)
class Stats:
  label: str
  unit: str
  minimum: float
  maximum: float
  mean: float
  p99: float


def decode_supply(raw: bytes) -> tuple[int, int, int]:
  """(bus millivolts, shunt milliamps, fault byte) from one 5-byte vendor IN frame."""
  if len(raw) != ASM_SUPPLY_LEN:
    raise ValueError(f"supply frame is {len(raw)} bytes, expected {ASM_SUPPLY_LEN}")
  return struct.unpack(ASM_SUPPLY_FMT, raw)


def percentile(values: Iterable[float], q: int) -> float:
  """Nearest-rank percentile: the smallest sample with at least q% of the run at or below it.

  Nearest rank rather than an interpolated one because every value it can return is a value the
  card actually drew. Interpolation invents a number between two samples, and for a peak-power
  question the invented number is always the optimistic one.
  """
  if not isinstance(q, int) or not 0 <= q <= 100:
    raise ValueError(f"percentile q must be an int in 0..100, got {q!r}")
  ordered = sorted(values)
  if not ordered:
    raise ValueError("percentile of an empty run is undefined")
  rank = -(-q * len(ordered) // 100)  # ceil(q*n/100) in exact integer arithmetic
  return ordered[max(0, rank - 1)]


def summarize(run: PhaseRun) -> list[Stats]:
  """min / max / mean / p99 of bus voltage, current and power over the run.

  Power is the per-sample product, never mean(V) * mean(A) -- voltage sags exactly when current
  peaks, so the product of the means understates the peak that matters.
  """
  if not run.samples:
    raise ValueError("no samples: there is nothing to summarise")
  columns = (("voltage", "V", [s.volts for s in run.samples]),
             ("current", "A", [s.amps for s in run.samples]),
             ("power", "W", [s.watts for s in run.samples]))
  return [Stats(label, unit, min(vals), max(vals), statistics.fmean(vals), percentile(vals, 99))
          for label, unit, vals in columns]


def sample_phase(phase: str, read_frame: Callable[[], bytes], rate_hz: float, duration_s: float, *,
                 now: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 reconnect_timeout_s: float = DEFAULT_RECONNECT_TIMEOUT_S,
                 progress: Callable[[int, int, Sample], None] | None = None) -> PhaseRun:
  """Sample at a fixed rate for a fixed duration, stopping dead on the first fault byte.

  Deadlines are absolute (t0 + i/rate) so a slow read does not compound into drift; the achieved
  rate is measured from the samples themselves rather than assumed from the request.
  """
  count = max(1, round(duration_s * rate_hz))
  interval = 1.0 / rate_hz
  samples: list[Sample] = []
  gaps: list[Gap] = []
  fault: Sample | None = None
  interrupted = link_lost = False
  down_since: float | None = None
  t0 = now()

  try:
    for i in range(count):
      delay = (t0 + i * interval) - now()
      if delay > 0:
        sleep(delay)
      t = now() - t0

      try:
        raw = read_frame()
      except SupplyReadError as e:
        # A gap is reported, never interpolated over. The bridge re-enumerates when another
        # process opens it, so a short gap is expected during coldstart -- a long one means the
        # link is gone and the rest of the window would be a fabrication.
        gaps.append(Gap(i, t, str(e)))
        if down_since is None:
          down_since = t
        if t - down_since >= reconnect_timeout_s:
          link_lost = True
          break
        continue
      down_since = None

      millivolts, milliamps, fault_byte = decode_supply(raw)
      sample = Sample(i, t, millivolts, milliamps, fault_byte, bytes(raw))
      samples.append(sample)
      if fault_byte:
        fault = sample
        break
      if progress is not None:
        progress(i, count, sample)
  except KeyboardInterrupt:
    # Caught here rather than around the call: the samples collected so far are the expensive
    # part of a ten-minute run and must still reach the CSV.
    interrupted = True

  return PhaseRun(phase=phase, requested_rate_hz=rate_hz, requested_duration_s=duration_s, samples=samples,
                  gaps=gaps, fault=fault, interrupted=interrupted, link_lost=link_lost)


class SupplyReader:
  """The INA231 behind the ASM2464, read without disturbing whatever is driving the bridge.

  See the module docstring: no claim, no detach, no reset, no set_configuration. Only
  libusb_open plus the vendor IN transfer on endpoint 0.
  """

  def __init__(self, timeout_ms: int = DEFAULT_USB_TIMEOUT_MS):
    self.timeout_ms = timeout_ms
    self.handle = None
    self.ids: tuple[int, int] | None = None
    self._buf = (ctypes.c_ubyte * ASM_SUPPLY_LEN)()
    self._libusb = None

  def _bindings(self):
    """tinygrad's libusb bindings and USB context, loaded lazily.

    Lazily so the statistics and the argument parsing in this file stay importable -- and
    self-testable -- on a machine with neither tinygrad nor a dock.
    """
    try:
      from tinygrad.runtime.autogen import libusb
      from tinygrad.runtime.support.usb import USB3
    except Exception as e:
      raise SupplyUnavailable("tinygrad's libusb bindings are not importable here: " + str(e)) from e
    try:
      ctx = USB3.ctx()
    except Exception as e:
      raise SupplyUnavailable("libusb did not initialise: " + str(e)) from e
    return libusb, ctx

  def open(self) -> None:
    libusb, ctx = self._bindings()
    self._libusb = libusb
    usb = _usb_helpers()
    for vendor, product in usb.CHESTNUT_USB_IDS:
      handle = libusb.libusb_open_device_with_vid_pid(ctx, vendor, product)
      if handle:
        self.handle, self.ids = handle, (vendor, product)
        return
    raise SupplyUnavailable(_no_dock_reason(usb))

  def read_frame(self) -> bytes:
    if self.handle is None:
      raise SupplyReadError("the bridge is not open")
    rc = self._transfer()
    if rc == ASM_SUPPLY_LEN:
      return bytes(self._buf)
    # One reopen and one retry: a process taking the bridge resets the device, which invalidates
    # this handle without meaning anything is wrong with the supply.
    first = self._strerror(rc)
    self.close()
    try:
      self.open()
    except SupplyUnavailable as e:
      raise SupplyReadError(f"{first}, and the bridge did not come back: {e}") from e
    rc = self._transfer()
    if rc == ASM_SUPPLY_LEN:
      return bytes(self._buf)
    raise SupplyReadError(f"{first}, and again after reopening: {self._strerror(rc)}")

  def _transfer(self) -> int:
    assert self._libusb is not None
    return self._libusb.libusb_control_transfer(self.handle, USB_CTRL_IN_VENDOR_DEVICE, ASM_SUPPLY_REQUEST,
                                                0, 0, self._buf, ASM_SUPPLY_LEN, self.timeout_ms)

  def _strerror(self, rc: int) -> str:
    if rc >= 0:
      return f"short read: {rc} of {ASM_SUPPLY_LEN} bytes"
    assert self._libusb is not None
    return f"libusb {rc}: " + ctypes.string_at(self._libusb.libusb_strerror(rc)).decode(errors="replace")

  def close(self) -> None:
    # The libusb context belongs to tinygrad's cached USB3.ctx(); closing our handle is ours to
    # do, exiting the context is not.
    if self.handle is not None and self._libusb is not None:
      self._libusb.libusb_close(self.handle)
    self.handle = None


def _usb_helpers():
  """openpilot's own chestnut USB ids and sysfs readers, loaded by file path.

  Importing the package would pull in HardwareBase -> cereal -> capnp; usb.py is stdlib-only.
  Loading the one file keeps the ids from drifting from the rest of openpilot without needing a
  built checkout, which is what `.elantra/probe_rdna2.py` does for the same reason.
  """
  import importlib.util
  path = REPO / "openpilot/common/hardware/usb.py"
  spec = importlib.util.spec_from_file_location("chestnut_usb", path)
  if spec is None or spec.loader is None:
    raise SupplyUnavailable("cannot load " + str(path))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _no_dock_reason(usb) -> str:
  """Why no custom-firmware chestnut opened, read from sysfs -- no USB traffic of its own."""
  for device in usb.usb_devices():
    ids = (usb.read_int(device / "idVendor", 16), usb.read_int(device / "idProduct", 16))
    if ids in usb.CHESTNUT_ROM_USB_IDS:
      return ("a chestnut is attached but running the stock ASMedia ROM, which is not the firmware tinygrad drives."
              + " Reflash it (system/hardware/chestnut/flash.py) -- the supply telemetry is a custom-firmware request")
  return "no chestnut found on the USB bus. Attach the dock and check it enumerates, then run .elantra/probe_rdna2.py"


def default_csv_path(phase: str, stamp: str) -> Path:
  """Off the openpilot checkout and onto persistent storage when there is any."""
  base = Path("/data/media/0/power") if Path("/data/media/0").is_dir() else Path.cwd()
  return base / f"power-{phase}-{stamp}.csv"


def write_csv(path: Path, run: PhaseRun) -> None:
  """Every sample and every gap, so a surprising summary can be re-examined without the card."""
  rows: list[tuple[int, list]] = []
  for s in run.samples:
    rows.append((s.index, [s.index, f"{s.t:.6f}", "sample", s.millivolts, s.milliamps, s.fault, f"{s.watts:.6f}",
                           s.raw.hex(), ""]))
  for g in run.gaps:
    rows.append((g.index, [g.index, f"{g.t:.6f}", "gap", "", "", "", "", "", g.detail]))
  rows.sort(key=lambda r: r[0])

  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w", newline="", encoding="utf-8") as f:
    f.write(f"# chestnut supply, phase={run.phase} (operator-asserted), started_utc={run.started_utc}\n")
    f.write(f"# requested_rate_hz={run.requested_rate_hz} requested_duration_s={run.requested_duration_s}\n")
    f.write(f"# samples={len(run.samples)} gaps={len(run.gaps)} fault={'yes' if run.fault else 'no'}\n")
    f.write("# watts = millivolts * milliamps / 1e6, per sample. raw is the 5-byte vendor IN frame, <HhB.\n")
    writer = csv.writer(f)
    writer.writerow(["index", "t_s", "event", "millivolts", "milliamps", "fault", "watts", "raw", "detail"])
    writer.writerows(row for _, row in rows)


def format_summary(run: PhaseRun, stats: list[Stats]) -> str:
  achieved = run.achieved_rate_hz
  lines = ["",
           "=" * 78,
           f"chestnut supply -- phase '{run.phase}' (operator-asserted)",
           "=" * 78,
           f"  samples            {len(run.samples)} over {run.elapsed_s:.2f} s",
           f"  requested rate     {run.requested_rate_hz:.1f} Hz",
           f"  achieved rate      {achieved:.2f} Hz" if achieved is not None else "  achieved rate      undefined (< 2 samples)",
           f"  longest interval   {run.max_interval_s * 1000:.1f} ms between consecutive samples",
           f"  read gaps          {len(run.gaps)}",
           f"  fault byte         {'SET -- see below' if run.fault else '0 on every sample'}",
           "",
           f"  {'quantity':<12}{'min':>12}{'max':>12}{'mean':>12}{'p99':>12}"]
  for s in stats:
    lines.append(f"  {s.label + ' (' + s.unit + ')':<12}{s.minimum:>12.3f}{s.maximum:>12.3f}{s.mean:>12.3f}{s.p99:>12.3f}")
  lines += ["",
            "  p99 is nearest-rank: a value the card actually drew, not an interpolated one.",
            "  The INA231 is configured AVG=1 / CT=1.1 ms and cannot see a sub-millisecond transient.",
            "  A scope and a DC current probe are still required -- .elantra/EGPU.md stage 3."]
  return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("phase", nargs="?", choices=sorted(PHASES), help="which of the four bench phases you are running")
  ap.add_argument("--duration", type=float, default=None, help="seconds to sample (default: per phase)")
  ap.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ, help=f"samples per second (default {DEFAULT_RATE_HZ:g})")
  ap.add_argument("--csv", type=Path, default=None, help="where to write the raw samples (default: alongside the run)")
  ap.add_argument("--reconnect-timeout", type=float, default=DEFAULT_RECONNECT_TIMEOUT_S,
                  help=f"seconds of unreadable bridge before giving up (default {DEFAULT_RECONNECT_TIMEOUT_S:g})")
  ap.add_argument("--self-test", action="store_true",
                  help="check the statistics against synthetic input and exit. Measures nothing.")
  return ap


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  ap = build_parser()
  args = ap.parse_args(argv)
  if args.self_test:
    return args
  if args.phase is None:
    ap.error("a phase is required: " + ", ".join(sorted(PHASES)) + " (or --self-test)")
  if args.rate <= 0:
    ap.error("--rate must be positive")
  if args.duration is None:
    args.duration = PHASES[args.phase].default_duration_s
  elif args.duration <= 0:
    ap.error("--duration must be positive")
  if args.reconnect_timeout <= 0:
    ap.error("--reconnect-timeout must be positive")
  return args


def self_test() -> int:
  """Exercise the statistics on synthetic input.

  THIS IS NOT A MEASUREMENT. It proves the arithmetic in this file is right; it says nothing
  whatever about the card, the dock or the supply, and no result it prints may be reported as a
  power figure. Only a real run against the hardware can do that.
  """
  checks: list[tuple[str, Callable[[], None]]] = []

  def case(name):
    def add(fn):
      checks.append((name, fn))
      return fn
    return add

  @case("decode round-trips a known frame")
  def _decode():
    assert decode_supply(struct.pack(ASM_SUPPLY_FMT, 12034, 1450, 0)) == (12034, 1450, 0)
    assert decode_supply(struct.pack(ASM_SUPPLY_FMT, 11800, -37, 0x04)) == (11800, -37, 0x04)

  @case("decode rejects a frame of the wrong length")
  def _decode_short():
    for bad in (b"", b"\x00" * 4, b"\x00" * 6):
      try:
        decode_supply(bad)
      except ValueError:
        continue
      raise AssertionError(f"{len(bad)} bytes must not decode")

  @case("nearest-rank percentile on a 1..100 ramp")
  def _pct():
    ramp = [float(v) for v in range(1, 101)]
    assert percentile(ramp, 0) == 1.0
    assert percentile(ramp, 50) == 50.0
    assert percentile(ramp, 99) == 99.0
    assert percentile(ramp, 100) == 100.0

  @case("p99 of a short run is the maximum, and says so by being it")
  def _pct_short():
    assert percentile([float(v) for v in range(1, 11)], 99) == 10.0
    assert percentile([7.5], 99) == 7.5

  @case("summary of a synthetic ramp")
  def _summary():
    triples = [(12000, 1000, 0), (12000, 2000, 0), (12000, 3000, 0), (12000, 4000, 0)]
    run = _synthetic_run(triples)
    stats = {s.label: s for s in summarize(run)}
    assert abs(stats["current"].mean - 2.5) < 1e-9, stats["current"].mean
    assert abs(stats["current"].maximum - 4.0) < 1e-9
    assert abs(stats["power"].mean - 30.0) < 1e-9, stats["power"].mean
    assert abs(stats["power"].maximum - 48.0) < 1e-9
    assert abs(stats["voltage"].minimum - 12.0) < 1e-9

  @case("power is the per-sample product, not the product of the means")
  def _power_pairing():
    stats = {s.label: s for s in summarize(_synthetic_run([(12000, 1000, 0), (10000, 5000, 0)]))}
    assert abs(stats["power"].mean - 31.0) < 1e-9, stats["power"].mean
    assert abs(stats["power"].maximum - 50.0) < 1e-9

  @case("summarising nothing raises instead of reporting zeros")
  def _empty():
    try:
      summarize(_synthetic_run([]))
    except ValueError:
      return
    raise AssertionError("an empty run must not summarise")

  @case("achieved rate is measured from the samples, not from the request")
  def _rate():
    run = _synthetic_run([(12000, 1000, 0)] * 101, rate=100.0, dt=0.02)
    assert run.achieved_rate_hz is not None and abs(run.achieved_rate_hz - 50.0) < 1e-6, run.achieved_rate_hz
    assert _synthetic_run([(12000, 1000, 0)]).achieved_rate_hz is None

  print("SELF-TEST -- statistics only. This is not a measurement: no card, dock or supply is")
  print("touched, and nothing printed here is a power figure. Only a real phase run produces one.")
  print("")
  failed = 0
  for name, fn in checks:
    try:
      fn()
    except Exception as e:
      failed += 1
      print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    else:
      print(f"  ok    {name}")
  print("")
  print(f"  {len(checks) - failed}/{len(checks)} arithmetic checks passed. Still not a measurement.")
  return EXIT_OK if failed == 0 else EXIT_FAULT


def _synthetic_run(triples, rate: float = 100.0, dt: float = 0.01) -> PhaseRun:
  """A PhaseRun built from literal numbers, for the self-test's arithmetic checks only."""
  samples = [Sample(i, i * dt, mv, ma, flt, struct.pack(ASM_SUPPLY_FMT, mv, ma, flt))
             for i, (mv, ma, flt) in enumerate(triples)]
  return PhaseRun("self-test", rate, len(samples) * dt, samples, [], None, False, False)


def _progress_printer(rate_hz: float) -> Callable[[int, int, Sample], None]:
  every = max(1, int(rate_hz * 5))

  def report(i: int, count: int, sample: Sample) -> None:
    # sample.t, not i / rate_hz: the elapsed time shown is the measured one, so a run that is
    # falling behind its schedule looks like it is falling behind rather than looking on time.
    if i % every == 0:
      print(f"  {sample.t:7.1f}s / {count / rate_hz:.0f}s   {sample.volts:6.3f} V  {sample.amps:6.3f} A  {sample.watts:7.2f} W")
  return report


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  if args.self_test:
    return self_test()

  phase = PHASES[args.phase]
  stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
  csv_path = args.csv if args.csv is not None else default_csv_path(phase.name, stamp)

  print("chestnut supply measurement -- port stage 8")
  print(f"  phase        {phase.name}")
  print(f"  you assert   {phase.precondition}")
  print(f"  sampling     {args.rate:g} Hz for {args.duration:g} s")
  print(f"  raw samples  {csv_path}")
  print("")

  reader = SupplyReader()
  try:
    reader.open()
  except SupplyUnavailable as e:
    print("cannot measure: " + str(e))
    return EXIT_NO_MEASUREMENT
  assert reader.ids is not None
  print(f"  bridge       {reader.ids[0]:#06x}:{reader.ids[1]:#06x}, endpoint 0 only -- not claimed, not reset")
  print("")

  try:
    run = sample_phase(phase.name, reader.read_frame, args.rate, args.duration,
                       reconnect_timeout_s=args.reconnect_timeout, progress=_progress_printer(args.rate))
  finally:
    reader.close()
  run.started_utc = stamp

  if not run.samples and not run.gaps:
    print("no samples and no gaps: nothing ran.")
    return EXIT_NO_MEASUREMENT

  write_csv(csv_path, run)
  if run.samples:
    print(format_summary(run, summarize(run)))
  print("")
  print(f"  raw samples written to {csv_path}")

  if run.fault is not None:
    f = run.fault
    print("")
    print("  FAULT -- the converter reported a fault byte. Sampling stopped there.")
    print(f"    sample index   {f.index}")
    print(f"    fault byte     {f.fault:#04x}")
    print(f"    raw frame      {f.raw.hex()}")
    print(f"    at             {f.t:.3f} s into the phase")
    print(f"    that sample    {f.volts:.3f} V, {f.amps:.3f} A, {f.watts:.2f} W")
    print("  This is an immediate fail. Do not average it away and do not put this card in a car")
    print("  until the supply is understood.")
    return EXIT_FAULT

  if run.link_lost:
    print("")
    print(f"  the bridge stopped answering for {args.reconnect_timeout:g} s and the run was abandoned.")
    print("  The samples above are real but the phase is incomplete -- this is not a result.")
    return EXIT_NO_MEASUREMENT

  if run.interrupted:
    print("")
    print("  interrupted. The samples above are real, but they do not cover the phase you asked for.")
    return EXIT_DEGRADED

  achieved = run.achieved_rate_hz
  if achieved is None or achieved < args.rate * RATE_SHORTFALL_FRACTION:
    print("")
    shortfall = f"{achieved:.2f} Hz" if achieved is not None else "undefined"
    print(f"  RATE SHORTFALL: asked for {args.rate:g} Hz, achieved {shortfall}.")
    print("  The samples are real and the CSV is good, but do not report these as a")
    print(f"  {args.rate:g} Hz peak -- at this rate the peak between samples was never looked at.")
    return EXIT_DEGRADED

  return EXIT_OK


if __name__ == "__main__":
  sys.exit(main())
