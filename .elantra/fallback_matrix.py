#!/usr/bin/env python3
"""
The eGPU fallback matrix, re-run against the device instead of against the source.

`.elantra/EGPU_SAFETY.md` section 4 lists eight failure cases and what each one is supposed to
do to the car. Rows 1 and 2 were exercised on hardware. Rows 3-6 are code-traced or covered by
a unit test. Rows 7 and 8 -- the dock leaving mid-drive, and the card faulting once it was
already driving the model -- have never been run at all.

The property the whole matrix exists to defend is two sentences long: an eGPU failure must
never stop the car engaging, and a mid-drive eGPU failure must degrade to the on-SoC model
cleanly. This runs the matrix against that property. For each row it states the precondition
it needs, how the failure is induced, and what it asserts afterwards -- in terms of state that
can actually be read off the device rather than inferred from the source: the params
`UsbGpuLoading`, `UsbGpuActive`, `EgpuDevice` and `EgpuDeviceDetected`, the two modeld gates
as modeld itself computes them, the `onroadEvents` selfdrived is publishing, and which model
is running.

**The refusals are the point.** A row whose precondition is not met on this device reports
`n/a`; a row that needs a moving car and a second person reports `needs-driver`; neither is
`pass`, and neither moves the exit code. A matrix that reports eight green rows when two of
them were never run is worse than no matrix, so no row here can go green without having
observed the state it claims. `--selftest` proves the assertions can fail.

This tool never induces a failure that needs hardware moved. It reads state and asserts on it;
the operator induces. That is deliberate -- a script cannot unplug a dock, and one that
pretended to would be reporting on nothing. Rows whose induced state is transient (5, 6, 7, 8)
are assessed by `--watch`, which observes while the operator induces.

Nothing here touches the GPU, the dock or the USB bus. It reads params, reads sysfs exactly
the way `usbgpu_present()` does, and subscribes to messaging. It never probes: `probe_once()`
opens a USBPCIDevice and takes an exclusive flock, which is not a thing to do beside a running
modeld.

Usage:
    python .elantra/fallback_matrix.py                    # what this device can prove right now
    python .elantra/fallback_matrix.py --watch 120        # watch an operator-induced failure
    python .elantra/fallback_matrix.py --watch 300 --with-driver   # rows 7 and 8, in the car
    python .elantra/fallback_matrix.py --selftest         # prove the assertions can fail
    python .elantra/fallback_matrix.py --json report.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

STOCK_MODELD = REPO / "openpilot/selfdrive/modeld/modeld.py"
SP_MODELD = REPO / "openpilot/sunnypilot/modeld_v2/modeld.py"

# What each runner's `USBGPU = ...` is expected to call. Compared against the real assignment
# so this tool can never assert a gate the car has stopped computing -- if modeld's expression
# changes, the rows that depend on it fail rather than quietly checking the old one.
STOCK_TERMS = frozenset({"usbgpu_present", "usbgpu_compiled", "egpu.enabled"})
SP_TERMS = frozenset({"usbgpu_present", "egpu.enabled"})

PASS, FAIL, NA, NEEDS_DRIVER = "pass", "FAIL", "n/a", "needs-driver"

BENCH = "offroad bench"
CAR_ON = "car on, onroad, stationary"
MOVING = "moving car, a second person driving"

# modelV2 may stop publishing briefly while modeld swaps the big model out for the resident
# small one. The swap is in-process, so this is generous; a modeld restart into a fresh 60 s
# load is an order of magnitude past it, which is the thing being told apart.
MODEL_GAP_MAX = 3.0

# Below this the car is not moving in any sense row 7 or 8 needs, whatever the operator says.
MOVING_MIN_MS = 4.0

# A load that failed inside a second or two is row 3 or row 4, not row 5's 60 s timeout.
LOAD_TIMEOUT_MIN = 30.0

EGPU_EVENTS = ("bigModelLoading", "bigModelFailed")


@dataclass(frozen=True)
class Check:
  ok: bool
  text: str
  detail: str = ""


@dataclass(frozen=True)
class Row:
  num: int
  case: str
  venue: str
  precondition: str
  induce: str
  asserts: str


@dataclass
class Result:
  row: Row
  status: str
  reason: str = ""
  checks: list[Check] = field(default_factory=list)


ROWS = (
  Row(
    num=1,
    case="No dock",
    venue=BENCH,
    precondition="no chestnut on the USB bus",
    induce="none -- the absence of the eGPU is the case",
    asserts="both modeld gates compute False, UsbGpuLoading is not latched, UsbGpuActive is unset, "
            + "no eGPU-sourced NO_ENTRY event, the on-SoC model is what runs",
  ),
  Row(
    num=2,
    case="Dock, unsupported card (RDNA2)",
    venue=BENCH,
    precondition="dock attached, and EgpuDevice or EgpuDeviceDetected naming a card asics.py blocklists",
    induce="none -- the card itself is the failure",
    asserts="egpu_build_ok() False (the build never attempts the eGPU target), detect.enabled() False, "
            + "both modeld gates False, no latch, the on-SoC model runs",
  ),
  Row(
    num=3,
    case="Dock, supported card, pkl missing",
    venue=CAR_ON,
    precondition="dock attached, a card AM supports, openpilot onroad",
    induce="operator removes the compiled big model and its .chunkmanifest, then restarts the car",
    asserts="UsbGpuActive is False (not unset -- the load was attempted and lost), UsbGpuLoading released, "
            + "modelV2 still publishing, chestnutState silent, no eGPU NO_ENTRY standing",
  ),
  Row(
    num=4,
    case="Dock, pkl corrupt or wrong vendor",
    venue=CAR_ON,
    precondition="dock attached, a card AM supports, openpilot onroad",
    induce="operator writes a .egpu marker naming the other vendor, or truncates the pkl, then restarts",
    asserts="the same end state as row 3 -- assert_pkl_matches raises inside the load, and the fallback it "
            + "lands in is the same one",
  ),
  Row(
    num=5,
    case="eGPU load times out (60 s)",
    venue=CAR_ON,
    precondition="dock attached, a card AM supports, openpilot onroad, --watch running across the load",
    induce="operator restarts the car with a card whose load hangs rather than raises. Not scriptable: "
           + "nothing this tool can do makes a healthy load hang",
    asserts="UsbGpuLoading held for >=30 s then released, UsbGpuActive False, bigModelFailed raised, "
            + "modelV2 publishing, no eGPU NO_ENTRY left latched",
  ),
  Row(
    num=6,
    case="modeld crash-loop",
    venue=CAR_ON,
    precondition="dock attached, a card AM supports, openpilot onroad, --watch running",
    induce="operator kills modeld repeatedly while the car is on and stationary",
    asserts="a NO_ENTRY is raised while modeld is down -- the car correctly cannot engage -- and when it "
            + "returns, modelV2 publishes again with no eGPU NO_ENTRY standing",
  ),
  Row(
    num=7,
    case="Dock unplugged mid-drive",
    venue=MOVING,
    precondition="the eGPU is driving the model (UsbGpuActive True), car onroad and actually moving",
    induce="a second person unplugs the chestnut while the driver drives. Never the person running this",
    asserts="chestnutPresent goes False, UsbGpuActive goes False, bigModelFailed raised (soft disable), "
            + "modelV2 keeps publishing, UsbGpuLoading not latched, no eGPU NO_ENTRY afterwards -- the car "
            + "can still engage on the small model",
  ),
  Row(
    num=8,
    case="eGPU faults after the model was driving",
    venue=MOVING,
    precondition="the eGPU is driving the model (UsbGpuActive True), car onroad and actually moving",
    induce="the card faults on its own, or a second person induces it. The dock stays present",
    asserts="chestnutPresent stays True while UsbGpuActive goes False through modeld's model.run() handler; "
            + "otherwise identical to row 7",
  ),
)


# ---------------------------------------------------------------------------- observation


@dataclass(frozen=True)
class Observed:
  available: bool = False
  unavailable_reason: str = ""
  dock: bool | None = None
  device_id: int | None = None
  device_source: str = ""
  asic_name: str = ""
  am_supported: bool | None = None
  vendor: str = ""
  assumed: bool | None = None
  enabled: bool | None = None
  build_ok: bool | None = None
  compiled: bool | None = None
  stock_gate: bool | None = None
  sp_gate: bool | None = None
  stock_terms: frozenset[str] = frozenset()
  sp_terms: frozenset[str] = frozenset()
  loading: bool | None = None
  active: bool | None = None
  egpu_device_raw: str = ""
  egpu_detected_raw: str = ""
  runner: str = ""
  messaging_reason: str = ""
  onroad: bool = False
  engageable: bool | None = None
  chestnut_present: bool | None = None
  model_alive: bool | None = None
  chestnut_state_alive: bool | None = None
  egpu_events: tuple[tuple[str, bool, bool], ...] = ()
  v_ego: float | None = None


def _dotted(node) -> str:
  if isinstance(node, ast.Attribute):
    return _dotted(node.value) + "." + node.attr
  if isinstance(node, ast.Name):
    return node.id
  return "?"


def usbgpu_terms(path: Path) -> frozenset[str]:
  """The functions a modeld's own `USBGPU = ...` calls.

  Read out of the source rather than assumed, so that a change to either runner's gate breaks
  the rows that depend on it instead of leaving this tool asserting a gate the car no longer
  computes. `.elantra/test_egpu.py` checks the same assignment for a different property.
  """
  tree = ast.parse(path.read_text(encoding="utf-8"))
  for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "USBGPU" for t in node.targets):
      return frozenset(_dotted(c.func) for c in ast.walk(node.value) if isinstance(c, ast.Call))
  return frozenset()


def _read_messaging(settle: float) -> dict:
  """One short settle window on the sockets selfdrived and modeld publish.

  Offroad, selfdriveState is simply not published -- that absence is how onroad is decided
  here, rather than by trusting a param that outlives the process which set it.
  """
  import openpilot.cereal.messaging as messaging
  services = ["deviceState", "selfdriveState", "modelV2", "onroadEvents", "carState", "chestnutState"]
  sm = messaging.SubMaster(services)
  end = time.monotonic() + settle
  while time.monotonic() < end:
    sm.update(100)

  events: tuple[tuple[str, bool, bool], ...] = ()
  if sm.seen["onroadEvents"]:
    events = tuple((str(e.name), bool(e.noEntry), bool(e.softDisable))
                   for e in sm["onroadEvents"] if str(e.name) in EGPU_EVENTS)
  return {
    "onroad": bool(sm.alive["selfdriveState"]),
    "engageable": bool(sm["selfdriveState"].engageable) if sm.seen["selfdriveState"] else None,
    "chestnut_present": bool(sm["deviceState"].chestnutPresent) if sm.seen["deviceState"] else None,
    "model_alive": bool(sm.alive["modelV2"]),
    "chestnut_state_alive": bool(sm.alive["chestnutState"]),
    "egpu_events": events,
    "v_ego": float(sm["carState"].vEgo) if sm.seen["carState"] else None,
  }


def observe(settle: float = 2.0) -> Observed:
  """Everything the matrix decides on, read once, through the shipping code paths."""
  try:
    from openpilot.common.params import Params
    from openpilot.selfdrive.modeld.helpers import usbgpu_compiled, usbgpu_present
    from openpilot.sunnypilot.egpu import detect, guard
    from openpilot.sunnypilot.models.helpers import get_active_bundle
  except Exception as e:
    # Not a swallowed error: off the device there is nothing to observe, and saying so is the
    # correct report. Every row below becomes n/a, which is exactly what it should be.
    return Observed(unavailable_reason=f"openpilot is not importable here ({type(e).__name__}: {e})")

  params = Params()
  dock = usbgpu_present()
  compiled = usbgpu_compiled()
  device_id = detect.resolve_device(params)
  vendor, assumed = detect.resolve(params)
  asic = detect.asic(params)
  enabled = detect.enabled(params)

  source = ""
  for key in ("EgpuDevice", "EgpuDeviceDetected"):
    if (params.get(key) or "").strip():
      source = key
      break

  bundle = get_active_bundle(params)
  runner = "stock (no sunnypilot bundle active)" if bundle is None else \
           f"sunnypilot bundle {getattr(bundle, 'display_name', '') or '?'}"

  live: dict = {}
  messaging_reason = ""
  try:
    live = _read_messaging(settle)
  except Exception as e:
    messaging_reason = f"messaging is not readable here ({type(e).__name__}: {e})"

  return Observed(
    available=True,
    dock=dock,
    device_id=device_id,
    device_source=source,
    asic_name=asic.name if asic is not None else "",
    am_supported=None if device_id is None else (asic is None or asic.am_supported),
    vendor=vendor,
    assumed=assumed,
    enabled=enabled,
    build_ok=guard.egpu_build_ok(params),
    compiled=compiled,
    stock_gate=dock and compiled and enabled,
    sp_gate=dock and enabled,
    stock_terms=usbgpu_terms(STOCK_MODELD),
    sp_terms=usbgpu_terms(SP_MODELD),
    loading=params.get("UsbGpuLoading"),
    active=params.get("UsbGpuActive"),
    egpu_device_raw=(params.get("EgpuDevice") or "").strip(),
    egpu_detected_raw=(params.get("EgpuDeviceDetected") or "").strip(),
    runner=runner,
    messaging_reason=messaging_reason,
    onroad=bool(live.get("onroad")),
    engageable=live.get("engageable"),
    chestnut_present=live.get("chestnut_present"),
    model_alive=live.get("model_alive"),
    chestnut_state_alive=live.get("chestnut_state_alive"),
    egpu_events=live.get("egpu_events", ()),
    v_ego=live.get("v_ego"),
  )


# ---------------------------------------------------------------------------- the watch


@dataclass(frozen=True)
class Watch:
  duration: float = 0.0
  samples: int = 0
  active_start: bool | None = None
  active_end: bool | None = None
  active_false_at: float | None = None
  loading_ever: bool = False
  loading_end: bool | None = None
  load_hold_s: float = 0.0
  chestnut_start: bool | None = None
  chestnut_end: bool | None = None
  chestnut_false_at: float | None = None
  big_failed_seen: bool = False
  egpu_no_entry_end: tuple[str, ...] = ()
  model_gap_s: float = 0.0
  model_alive_end: bool = False
  engageable_end: bool | None = None
  v_ego_max: float = 0.0
  modeld_down_seen: bool = False
  modeld_running_end: bool | None = None
  no_entry_while_down: tuple[str, ...] = ()


def watch_device(duration: float, echo=print) -> tuple[Watch | None, str]:
  """Sample the state the matrix asserts on, for as long as the operator needs to induce.

  What is recorded here is transitions, not levels. Rows 5-8 are each about something that
  happens and is then over: a snapshot taken afterwards cannot tell a clean mid-drive fallback
  from a card that was never used, because both end with UsbGpuActive False and a small model
  driving.
  """
  try:
    import openpilot.cereal.messaging as messaging
    from openpilot.common.params import Params
  except Exception as e:
    return None, f"messaging is not available here ({type(e).__name__}: {e})"

  params = Params()
  sm = messaging.SubMaster(["deviceState", "selfdriveState", "modelV2", "onroadEvents", "carState", "managerState"])

  t0 = time.monotonic()
  samples = 0
  active_start = active_end = chestnut_start = chestnut_end = None
  engageable_end = modeld_running_end = None
  active_false_at = chestnut_false_at = None
  load_first = load_last = None
  loading_ever = False
  loading_end = None
  big_failed_seen = False
  egpu_no_entry_end: tuple[str, ...] = ()
  model_gap = 0.0
  last_model_alive = 0.0
  model_alive_end = False
  v_ego_max = 0.0
  modeld_down_seen = False
  no_entry_while_down: set[str] = set()

  while (t := time.monotonic() - t0) < duration:
    sm.update(100)
    samples += 1

    active = params.get("UsbGpuActive")
    loading = params.get("UsbGpuLoading")
    if active_start is None:
      active_start = active
    active_end = active
    if active is False and active_false_at is None:
      active_false_at = t
    if loading is True:
      loading_ever = True
      load_first = t if load_first is None else load_first
      load_last = t
    loading_end = loading

    if sm.seen["deviceState"]:
      present = bool(sm["deviceState"].chestnutPresent)
      if chestnut_start is None:
        chestnut_start = present
      chestnut_end = present
      if not present and chestnut_false_at is None:
        chestnut_false_at = t

    if sm.seen["selfdriveState"]:
      engageable_end = bool(sm["selfdriveState"].engageable)
    if sm.seen["carState"]:
      v_ego_max = max(v_ego_max, float(sm["carState"].vEgo))

    modeld_running = None
    if sm.seen["managerState"]:
      procs = sm["managerState"].processes
      if any(p.name == "modeld" and p.shouldBeRunning for p in procs):
        modeld_running = any(p.name == "modeld" and p.running for p in procs)
        modeld_running_end = modeld_running
        modeld_down_seen = modeld_down_seen or not modeld_running

    if sm.updated["onroadEvents"]:
      no_entry = {str(e.name) for e in sm["onroadEvents"] if e.noEntry}
      big_failed_seen = big_failed_seen or any(str(e.name) == "bigModelFailed" for e in sm["onroadEvents"])
      egpu_no_entry_end = tuple(sorted(n for n in no_entry if n in EGPU_EVENTS))
      if modeld_running is False:
        no_entry_while_down |= no_entry

    alive = bool(sm.alive["modelV2"])
    model_alive_end = alive
    if alive:
      last_model_alive = t
    else:
      model_gap = max(model_gap, t - last_model_alive)

    if echo is not None and samples % 50 == 0:
      echo(f"  ..    {t:5.1f}s  UsbGpuActive={active!r} UsbGpuLoading={loading!r} "
           + f"chestnut={chestnut_end!r} modelV2={'alive' if alive else 'DOWN'} vEgoMax={v_ego_max:.1f}")

  watched = Watch(
    duration=duration,
    samples=samples,
    active_start=active_start,
    active_end=active_end,
    active_false_at=active_false_at,
    loading_ever=loading_ever,
    loading_end=loading_end,
    load_hold_s=(load_last - load_first) if load_first is not None and load_last is not None else 0.0,
    chestnut_start=chestnut_start,
    chestnut_end=chestnut_end,
    chestnut_false_at=chestnut_false_at,
    big_failed_seen=big_failed_seen,
    egpu_no_entry_end=egpu_no_entry_end,
    model_gap_s=model_gap,
    model_alive_end=model_alive_end,
    engageable_end=engageable_end,
    v_ego_max=v_ego_max,
    modeld_down_seen=modeld_down_seen,
    modeld_running_end=modeld_running_end,
    no_entry_while_down=tuple(sorted(no_entry_while_down)),
  )
  return watched, ""


# ---------------------------------------------------------------------------- assertions
#
# Pure, so --selftest can prove each one fails on the state it is supposed to reject. An
# assertion that cannot fail is the same lie as a row that was never run.


def model_description(obs: Observed) -> str:
  if obs.active is True:
    return "the big model, on the eGPU (UsbGpuActive True)"
  if obs.active is False:
    return "the on-SoC small model, after an eGPU fallback (UsbGpuActive False)"
  return "the on-SoC small model (UsbGpuActive unset -- the eGPU path was never entered)"


def gate_model_checks(obs: Observed) -> list[Check]:
  return [
    Check(obs.stock_terms == STOCK_TERMS, "the stock runner's gate is the one this tool models",
          f"modeld.py computes USBGPU from {sorted(obs.stock_terms)}"),
    Check(obs.sp_terms == SP_TERMS, "the sunnypilot runner's gate is the one this tool models",
          f"modeld_v2/modeld.py computes USBGPU from {sorted(obs.sp_terms)}"),
  ]


def no_latch_checks(obs: Observed) -> list[Check]:
  """The state that says the car was never held out of engagement by the eGPU.

  `UsbGpuActive` unset and `UsbGpuActive` False are different answers: selfdrived reads False
  as `big_failed` and raises bigModelFailed, so an unused eGPU path that left False behind
  would soft-disable a car which never had a big model in the first place.
  """
  return [
    Check(obs.loading is not True, "UsbGpuLoading is not latched", f"UsbGpuLoading={obs.loading!r}"),
    Check(obs.active is None, "UsbGpuActive is unset, so selfdrived sees no big-model failure",
          f"UsbGpuActive={obs.active!r}"),
  ]


def engagement_checks(obs: Observed) -> list[Check]:
  """Nothing the eGPU did is standing between the driver and engagement.

  Deliberately not `engageable is True`: engageable is False for a dozen reasons on a parked
  car -- calibration, a door, no panda -- and a row that failed on those would be noise. The
  claim this matrix is entitled to make is the narrow one, so that is the one it asserts, with
  engageable reported alongside as evidence rather than as a gate.
  """
  if not obs.onroad:
    return [Check(True, "selfdrived is not running (offroad), so engagement is asserted through the params it reads",
                  "an onroad run adds selfdriveState.engageable and the live onroadEvents")]
  blocking = tuple(name for name, no_entry, _ in obs.egpu_events if no_entry)
  return [
    Check(not blocking, "no eGPU-sourced NO_ENTRY event is present", f"blocking: {list(blocking)}"),
    Check(True, f"selfdriveState.engageable is {obs.engageable!r}",
          "reported, not asserted -- it moves for reasons unrelated to the eGPU"),
  ]


def assess_row1(obs: Observed) -> list[Check]:
  return [
    Check(obs.dock is False, "no chestnut is on the USB bus", f"usbgpu_present()={obs.dock!r}"),
    Check(obs.stock_gate is False, "the stock runner's USBGPU is False", f"stock gate={obs.stock_gate!r}"),
    Check(obs.sp_gate is False, "the sunnypilot runner's USBGPU is False", f"sunnypilot gate={obs.sp_gate!r}"),
    *gate_model_checks(obs),
    *no_latch_checks(obs),
    *engagement_checks(obs),
  ]


def assess_row2(obs: Observed) -> list[Check]:
  card = f"device 0x{obs.device_id:04x} ({obs.asic_name}) from {obs.device_source}" if obs.device_id is not None else "no device id"
  return [
    Check(obs.dock is True, "the chestnut is attached", f"usbgpu_present()={obs.dock!r}"),
    Check(obs.am_supported is False, "the gate can see the card, and AM refuses it", card),
    Check(obs.build_ok is False, "egpu_build_ok() is False, so SCons never attempts the eGPU target",
          f"egpu_build_ok()={obs.build_ok!r}"),
    Check(obs.enabled is False, "detect.enabled() is False", f"enabled()={obs.enabled!r}"),
    Check(obs.stock_gate is False, "the stock runner's USBGPU is False", f"stock gate={obs.stock_gate!r}"),
    Check(obs.sp_gate is False, "the sunnypilot runner's USBGPU is False", f"sunnypilot gate={obs.sp_gate!r}"),
    *gate_model_checks(obs),
    *no_latch_checks(obs),
    *engagement_checks(obs),
  ]


def assess_load_fallback(obs: Observed) -> list[Check]:
  """Rows 3 and 4: the eGPU load was attempted, lost, and the car carried on without it."""
  blocking = tuple(name for name, no_entry, _ in obs.egpu_events if no_entry)
  return [
    Check(obs.active is False, "UsbGpuActive is False -- the load was attempted and did not take",
          f"UsbGpuActive={obs.active!r}"),
    Check(obs.loading is not True, "UsbGpuLoading was released", f"UsbGpuLoading={obs.loading!r}"),
    Check(obs.model_alive is True, "modelV2 is publishing, so a model is driving", f"modelV2 alive={obs.model_alive!r}"),
    Check(obs.chestnut_state_alive is not True, "chestnutState is silent, so the eGPU is not the one running",
          f"chestnutState alive={obs.chestnut_state_alive!r}"),
    Check(not blocking, "no eGPU-sourced NO_ENTRY event is present", f"blocking: {list(blocking)}"),
    Check(True, f"selfdriveState.engageable is {obs.engageable!r}", "reported, not asserted"),
  ]


def assess_degradation(w: Watch, require_dock_loss: bool) -> list[Check]:
  """Rows 7 and 8, and the shape row 5 lands in: the eGPU stopped and the car did not.

  `require_dock_loss` is the only difference between row 7 and row 8. Row 7's trigger is the
  dock disappearing, which reaches selfdrived through `chestnutPresent`; row 8's is the card
  faulting with the dock still there, which reaches modeld through the `model.run()` handler.
  Asserting the trigger keeps one from being reported as evidence for the other.
  """
  checks = [
    Check(w.active_start is True, "the eGPU was driving the model when the watch started",
          f"UsbGpuActive at t=0: {w.active_start!r}"),
    Check(w.active_false_at is not None, "UsbGpuActive went False -- modeld handed over to the small model",
          f"at t={w.active_false_at:.1f}s" if w.active_false_at is not None else "never observed"),
    Check(w.big_failed_seen, "bigModelFailed was raised, so the driver was told",
          "soft disable plus a permanent alert"),
    Check(w.loading_end is not True, "UsbGpuLoading is not latched after the failure",
          f"UsbGpuLoading={w.loading_end!r}"),
    Check(w.model_alive_end, "modelV2 is still publishing at the end of the watch", "the small model took over"),
    Check(w.model_gap_s <= MODEL_GAP_MAX, "modelV2 never stopped for longer than the swap should take",
          f"longest gap {w.model_gap_s:.1f}s, limit {MODEL_GAP_MAX:.1f}s"),
    Check(not w.egpu_no_entry_end, "no eGPU-sourced NO_ENTRY is standing afterwards -- the car can re-engage",
          f"standing: {list(w.egpu_no_entry_end)}"),
    Check(True, f"selfdriveState.engageable at the end: {w.engageable_end!r}", "reported, not asserted"),
  ]
  if require_dock_loss:
    trigger = Check(w.chestnut_false_at is not None, "chestnutPresent went False -- the dock actually left",
                    f"at t={w.chestnut_false_at:.1f}s" if w.chestnut_false_at is not None else "the dock never left")
  else:
    trigger = Check(w.chestnut_end is True, "the dock stayed present -- this is the card faulting, not an unplug",
                    f"chestnutPresent={w.chestnut_end!r}")
  checks.insert(1, trigger)
  return checks


def assess_crash_loop(w: Watch) -> list[Check]:
  """Row 6. The one row whose correct answer is that the car cannot engage."""
  return [
    Check(w.modeld_down_seen, "modeld was observed down", "managerState reported it not running while it should be"),
    Check(bool(w.no_entry_while_down),
          "a NO_ENTRY was raised while modeld was down -- the car correctly refused to engage",
          f"events: {list(w.no_entry_while_down)}"),
    Check(w.modeld_running_end is True, "modeld came back", f"running at the end: {w.modeld_running_end!r}"),
    Check(w.model_alive_end, "modelV2 is publishing again", ""),
    Check(not w.egpu_no_entry_end, "no eGPU-sourced NO_ENTRY is standing afterwards",
          f"standing: {list(w.egpu_no_entry_end)}"),
  ]


# ---------------------------------------------------------------------------- preconditions


def supported_card_reason(obs: Observed) -> str | None:
  """Why this device is not a supported-card bench, or None if it is."""
  if not obs.dock:
    return "no chestnut is attached"
  if obs.device_id is None:
    return ("the gate cannot see the card: EgpuDevice is unset and EgpuDeviceDetected is empty. probe_once() "
            + "cannot enumerate the bridge on a cold boot (EGPU_SAFETY.md section 4), so set EgpuDevice before "
            + "running any row that turns on the card's identity")
  if obs.am_supported is False:
    return f"the attached card (0x{obs.device_id:04x}, {obs.asic_name}) is one tinygrad's AM driver refuses"
  return None


def unsupported_card_reason(obs: Observed) -> str | None:
  """Why this device is not an unsupported-card bench, or None if it is."""
  if not obs.dock:
    return "no chestnut is attached, so there is no card to be unsupported"
  if obs.device_id is None:
    return ("the gate cannot see the card: EgpuDevice is unset and EgpuDeviceDetected is empty, so enabled() reads "
            + "the card as supported by default. Set EgpuDevice (0x73ff for the 6600 XT) to make this row runnable")
  if obs.am_supported is not False:
    return f"the attached card (0x{obs.device_id:04x}) is not on the asics.py blocklist"
  return None


def _msg(obs: Observed) -> str:
  return f" ({obs.messaging_reason})" if obs.messaging_reason else ""


def finish(row: Row, checks: list[Check]) -> Result:
  return Result(row, FAIL if any(not c.ok for c in checks) else PASS, checks=checks)


def run_row(row: Row, obs: Observed, w: Watch | None, watch_reason: str, with_driver: bool) -> Result:
  """One row's verdict. Every path out of here is pass, FAIL, n/a or needs-driver -- there is
  no path that reports a row as established without having read the state it claims."""
  # Ahead of everything else, including "is there even a device here". A row that needs a
  # moving car and a second person should say so on every run, not only on the runs where the
  # rest of the environment happened to be interesting.
  if row.num in (7, 8) and not with_driver:
    return Result(row, NEEDS_DRIVER,
                  "this row needs a moving car and a second person to induce the failure. Re-run it in the car "
                  + "with --watch and --with-driver. Refusing, rather than reporting a row that was never run")

  if not obs.available:
    return Result(row, NA, obs.unavailable_reason)

  if row.num == 1:
    if obs.dock:
      return Result(row, NA, "a chestnut is attached; unplug it to run row 1")
    return finish(row, assess_row1(obs))

  if row.num == 2:
    if (why := unsupported_card_reason(obs)) is not None:
      return Result(row, NA, why)
    return finish(row, assess_row2(obs))

  if (why := supported_card_reason(obs)) is not None:
    return Result(row, NA, why)

  if row.num in (3, 4):
    if not obs.onroad:
      return Result(row, NA, "openpilot is offroad; this row needs modeld to have attempted the load" + _msg(obs))
    if obs.active is None:
      return Result(row, NA, "UsbGpuActive is unset: modeld never attempted an eGPU load, so the induced state "
                             + "is not present")
    return finish(row, assess_load_fallback(obs))

  # Rows 5-8 are transitions. A snapshot taken afterwards cannot tell them apart, or tell any
  # of them from a card that was simply never used, so they are only ever read from a watch.
  if w is None:
    return Result(row, NA, watch_reason or "no watch was run; pass --watch SECONDS with the operator ready to induce")

  if row.num == 5:
    if not w.loading_ever:
      return Result(row, NA, "UsbGpuLoading was never observed True during the watch, so no load was watched")
    if w.load_hold_s < LOAD_TIMEOUT_MIN:
      return Result(row, NA, f"the load was held for only {w.load_hold_s:.1f}s -- that is a fast failure "
                             + f"(row 3 or 4), not the {LOAD_TIMEOUT_MIN:.0f}s+ timeout row 5 is about")
    return finish(row, assess_degradation(w, require_dock_loss=False))

  if row.num == 6:
    if not w.modeld_down_seen:
      return Result(row, NA, "modeld was never observed down during the watch")
    return finish(row, assess_crash_loop(w))

  if w.active_start is not True:
    return Result(row, NA, "the eGPU was not driving the model when the watch started "
                           + f"(UsbGpuActive={w.active_start!r}), so there was nothing to degrade from")
  if w.v_ego_max < MOVING_MIN_MS:
    return Result(row, NA, f"the car never moved during the watch (max vEgo {w.v_ego_max:.1f} m/s, floor "
                           + f"{MOVING_MIN_MS:.1f}). A stationary observation does not close this row")
  if row.num == 7:
    if w.chestnut_false_at is None:
      return Result(row, NA, "the dock never disappeared during the watch, so row 7 was not induced")
    return finish(row, assess_degradation(w, require_dock_loss=True))

  if w.chestnut_end is not True or w.active_false_at is None:
    return Result(row, NA, "no card fault with the dock still present was observed during the watch")
  return finish(row, assess_degradation(w, require_dock_loss=False))


# ---------------------------------------------------------------------------- report


def print_environment(obs: Observed) -> None:
  print("=" * 100)
  print("Device state")
  print("=" * 100)
  if not obs.available:
    print("  --    " + obs.unavailable_reason)
    print("  --    every row below is n/a: this matrix only means something run on the device")
    return
  card = "unknown" if obs.device_id is None else f"0x{obs.device_id:04x} {obs.asic_name or '(not blocklisted)'}"
  print(f"  dock (usbgpu_present)   {obs.dock}")
  print(f"  card                    {card}  [EgpuDevice={obs.egpu_device_raw or '<unset>'}, "
        + f"EgpuDeviceDetected={obs.egpu_detected_raw or '<unset>'}]")
  print(f"  vendor                  {obs.vendor}{' (assumed)' if obs.assumed else ''}")
  print(f"  am_supports             {obs.am_supported}")
  print(f"  detect.enabled()        {obs.enabled}")
  print(f"  guard.egpu_build_ok()   {obs.build_ok}")
  print(f"  usbgpu_compiled()       {obs.compiled}")
  print(f"  modeld gates            stock={obs.stock_gate}  sunnypilot={obs.sp_gate}")
  print(f"  UsbGpuLoading           {obs.loading!r}")
  print(f"  UsbGpuActive            {obs.active!r}")
  print(f"  modeld runner           {obs.runner}")
  print(f"  model running           {model_description(obs)}")
  if obs.messaging_reason:
    print(f"  messaging               {obs.messaging_reason}")
    return
  print(f"  onroad                  {obs.onroad}  (selfdriveState {'alive' if obs.onroad else 'not published'})")
  print(f"  engageable              {obs.engageable!r}")
  print(f"  chestnutPresent         {obs.chestnut_present!r}")
  print(f"  modelV2 / chestnutState {'alive' if obs.model_alive else 'not alive'} / "
        + f"{'alive' if obs.chestnut_state_alive else 'not alive'}")
  print(f"  eGPU events             {[n for n, _, _ in obs.egpu_events] or 'none'}")


def print_result(r: Result) -> None:
  print("")
  print("-" * 100)
  print(f"Row {r.row.num} -- {r.row.case}".ljust(84) + f"[ {r.status} ]".rjust(16))
  print("-" * 100)
  print(f"  venue         {r.row.venue}")
  print(f"  precondition  {r.row.precondition}")
  print(f"  induce        {r.row.induce}")
  print(f"  asserts       {r.row.asserts}")
  if r.reason:
    print(f"  {'refused' if r.status == NEEDS_DRIVER else 'not run here':<13} {r.reason}")
  for c in r.checks:
    print(f"  {'ok   ' if c.ok else 'FAIL '} {c.text}" + (f"  [{c.detail}]" if c.detail else ""))


def print_summary(results: list[Result]) -> None:
  counts = {s: sum(1 for r in results if r.status == s) for s in (PASS, NA, NEEDS_DRIVER, FAIL)}
  print("")
  print("=" * 100)
  print("Summary")
  print("=" * 100)
  for status in (PASS, NA, NEEDS_DRIVER, FAIL):
    print(f"  {status:<14} {counts[status]}")
  print("")
  for r in results:
    print(f"  row {r.row.num}  {r.status:<13} {r.row.case}")

  not_run = [r.row.num for r in results if r.status in (NA, NEEDS_DRIVER)]
  if not_run:
    print("")
    print(f"  Rows not exercised by this run: {', '.join(str(n) for n in not_run)}.")
    print("  They are not passing. They were not run.")
  driver = [r.row.num for r in results if r.status == NEEDS_DRIVER]
  if driver:
    print(f"  Rows {', '.join(str(n) for n in driver)} need a moving car and a second person. Until someone")
    print("  drives them, the mid-drive degradation path has never been observed on hardware.")


# ---------------------------------------------------------------------------- selftest


def selftest() -> int:
  """Prove every assertion can fail.

  A check that cannot fail is the same lie as a row that was never run, so each assessor is
  fed both the state it should accept and the states it must reject -- and the two refusals
  that matter most (no driver, a car that never moved) are checked through `run_row` itself.
  """
  failures: list[str] = []

  def case(name: str, got, want) -> None:
    if got == want:
      print("  ok    " + name)
    else:
      failures.append(name)
      print(f"  FAIL  {name}: got {got!r}, want {want!r}")

  def verdict(checks: list[Check]) -> str:
    return FAIL if any(not c.ok for c in checks) else PASS

  print("\n[row 1] no dock")
  clean = Observed(available=True, dock=False, stock_gate=False, sp_gate=False, stock_terms=STOCK_TERMS,
                   sp_terms=SP_TERMS, loading=False, active=None)
  case("a clean no-dock device passes", verdict(assess_row1(clean)), PASS)
  case("a latched UsbGpuLoading fails", verdict(assess_row1(replace(clean, loading=True))), FAIL)
  case("a stale UsbGpuActive=False fails", verdict(assess_row1(replace(clean, active=False))), FAIL)
  case("a gate that no longer matches modeld fails",
       verdict(assess_row1(replace(clean, stock_terms=frozenset({"usbgpu_present"})))), FAIL)
  case("an eGPU NO_ENTRY onroad fails",
       verdict(assess_row1(replace(clean, onroad=True, egpu_events=(("bigModelLoading", True, False),)))), FAIL)
  case("an unrelated NO_ENTRY onroad does not fail this row",
       verdict(assess_row1(replace(clean, onroad=True, engageable=False))), PASS)

  print("\n[row 2] dock, blocklisted card")
  rdna2 = Observed(available=True, dock=True, device_id=0x73FF, device_source="EgpuDevice",
                   asic_name="Navi 23 [Radeon RX 6600/6600 XT/6600M]", am_supported=False, enabled=False,
                   build_ok=False, compiled=False, stock_gate=False, sp_gate=False, stock_terms=STOCK_TERMS,
                   sp_terms=SP_TERMS, loading=False, active=None)
  case("the blocklisted card passes", verdict(assess_row2(rdna2)), PASS)
  case("a build gate that says yes fails", verdict(assess_row2(replace(rdna2, build_ok=True))), FAIL)
  case("enabled() saying yes fails", verdict(assess_row2(replace(rdna2, enabled=True))), FAIL)
  case("either runner gate being True fails", verdict(assess_row2(replace(rdna2, sp_gate=True))), FAIL)

  print("\n[rows 3-4] the load was attempted and lost")
  lost = Observed(available=True, onroad=True, active=False, loading=False, model_alive=True,
                  chestnut_state_alive=False, engageable=True)
  case("a clean load fallback passes", verdict(assess_load_fallback(lost)), PASS)
  case("a latched loading flag fails", verdict(assess_load_fallback(replace(lost, loading=True))), FAIL)
  case("modelV2 not publishing fails", verdict(assess_load_fallback(replace(lost, model_alive=False))), FAIL)
  case("a standing eGPU NO_ENTRY fails",
       verdict(assess_load_fallback(replace(lost, egpu_events=(("bigModelFailed", True, True),)))), FAIL)

  print("\n[rows 7-8] mid-drive degradation")
  unplug = Watch(duration=120.0, samples=1200, active_start=True, active_end=False, active_false_at=41.0,
                 loading_end=False, chestnut_start=True, chestnut_end=False, chestnut_false_at=40.5,
                 big_failed_seen=True, model_gap_s=0.4, model_alive_end=True, engageable_end=True, v_ego_max=18.0)
  case("a clean unplug passes row 7", verdict(assess_degradation(unplug, require_dock_loss=True)), PASS)
  case("UsbGpuActive never clearing fails", verdict(assess_degradation(replace(unplug, active_false_at=None), True)), FAIL)
  case("no bigModelFailed fails", verdict(assess_degradation(replace(unplug, big_failed_seen=False), True)), FAIL)
  case("a latched loading flag fails", verdict(assess_degradation(replace(unplug, loading_end=True), True)), FAIL)
  case("modelV2 gone at the end fails", verdict(assess_degradation(replace(unplug, model_alive_end=False), True)), FAIL)
  case("a long modelV2 gap fails", verdict(assess_degradation(replace(unplug, model_gap_s=61.0), True)), FAIL)
  case("a standing eGPU NO_ENTRY fails",
       verdict(assess_degradation(replace(unplug, egpu_no_entry_end=("bigModelLoading",)), True)), FAIL)
  case("the dock never leaving fails row 7", verdict(assess_degradation(replace(unplug, chestnut_false_at=None), True)), FAIL)
  case("an unplug cannot be read as row 8", verdict(assess_degradation(unplug, require_dock_loss=False)), FAIL)
  fault = replace(unplug, chestnut_end=True, chestnut_false_at=None)
  case("a card fault with the dock present passes row 8", verdict(assess_degradation(fault, require_dock_loss=False)), PASS)
  case("a card fault cannot be read as row 7", verdict(assess_degradation(fault, require_dock_loss=True)), FAIL)

  print("\n[row 6] crash loop")
  loop = Watch(modeld_down_seen=True, no_entry_while_down=("processNotRunning",), modeld_running_end=True,
               model_alive_end=True)
  case("a crash loop that recovered passes", verdict(assess_crash_loop(loop)), PASS)
  case("no NO_ENTRY while down fails", verdict(assess_crash_loop(replace(loop, no_entry_while_down=()))), FAIL)
  case("modeld still down at the end fails", verdict(assess_crash_loop(replace(loop, modeld_running_end=False))), FAIL)

  print("\n[refusals] the states that must never report pass")
  blocklisted = Observed(available=True, dock=True, device_id=0x73FF, am_supported=False, asic_name="Navi 23")
  supported = Observed(available=True, dock=True, device_id=0x744C, am_supported=True, onroad=True)
  stationary = Watch(active_start=True, active_end=False, active_false_at=1.0, chestnut_start=True, chestnut_end=False,
                     chestnut_false_at=0.9, big_failed_seen=True, loading_end=False, model_alive_end=True, v_ego_max=0.0)
  moving = replace(stationary, v_ego_max=18.0)
  for num in (7, 8):
    row = next(r for r in ROWS if r.num == num)
    case(f"row {num} refuses without --with-driver", run_row(row, supported, moving, "", False).status, NEEDS_DRIVER)
    case(f"row {num} refuses with a driver but no watch", run_row(row, supported, None, "", True).status, NA)
    case(f"row {num} on a stationary car is n/a, not pass",
         run_row(row, supported, stationary, "", True).status, NA)
  case("row 7 with a driver, moving, and a real unplug does pass",
       run_row(next(r for r in ROWS if r.num == 7), supported, moving, "", True).status, PASS)
  case("row 3 on a blocklisted card is n/a", run_row(next(r for r in ROWS if r.num == 3), blocklisted, None, "", False).status, NA)
  case("row 2 with no dock is n/a", run_row(next(r for r in ROWS if r.num == 2), Observed(available=True, dock=False), None, "", False).status, NA)
  case("row 2 with a dock but a blind gate is n/a",
       run_row(next(r for r in ROWS if r.num == 2), Observed(available=True, dock=True), None, "", False).status, NA)
  off_device = Observed(unavailable_reason="not on the device")
  case("off the device every row is n/a rather than passing",
       {run_row(r, off_device, None, "", True).status for r in ROWS}, {NA})
  case("off the device, rows 7 and 8 still refuse for want of a driver",
       {run_row(r, off_device, None, "", False).status for r in ROWS if r.num in (7, 8)}, {NEEDS_DRIVER})

  print("\n[gate model] the terms this tool asserts are the ones modeld computes")
  for path, want, name in ((STOCK_MODELD, STOCK_TERMS, "stock"), (SP_MODELD, SP_TERMS, "sunnypilot")):
    if path.is_file():
      case(f"the {name} runner's USBGPU assignment still calls {sorted(want)}", usbgpu_terms(path), want)
    else:
      print(f"  --    {name} modeld not in this tree ({path}); the gate-model check needs the repo")

  print("")
  if failures:
    print(f"{len(failures)} selftest failure(s): " + ", ".join(failures))
    return 1
  print("selftest: every assertion rejects the state it is meant to reject")
  return 0


# ---------------------------------------------------------------------------- main


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--watch", type=float, default=0.0, metavar="SECONDS",
                  help="observe while an operator induces a failure; rows 5-8 are only readable from a watch")
  ap.add_argument("--with-driver", action="store_true",
                  help="attest that a second person is driving. Without it, rows 7 and 8 refuse to run")
  ap.add_argument("--settle", type=float, default=2.0, help="seconds to let the messaging sockets settle (default 2)")
  ap.add_argument("--json", default=None, metavar="PATH", help="also write the report as JSON")
  ap.add_argument("--selftest", action="store_true", help="check that the assertions can fail, then exit")
  args = ap.parse_args()

  if args.selftest:
    print("eGPU fallback matrix -- selftest")
    return selftest()

  print("eGPU fallback matrix")
  print("  repo:   " + str(REPO))
  print("  matrix: .elantra/EGPU_SAFETY.md section 4")
  obs = observe(args.settle)
  print_environment(obs)

  w, watch_reason = None, ""
  if args.watch > 0:
    print("")
    print("=" * 100)
    print(f"Watching for {args.watch:.0f}s -- induce the failure now")
    print("=" * 100)
    w, watch_reason = watch_device(args.watch)
    if w is None:
      print("  --    " + watch_reason)
  elif args.with_driver:
    print("")
    print("  --    --with-driver was passed without --watch. Rows 7 and 8 are transitions: they need a watch")
    print("        running while the failure is induced, so they will still refuse.")

  results = [run_row(row, obs, w, watch_reason, args.with_driver) for row in ROWS]
  for r in results:
    print_result(r)
  print_summary(results)

  if args.json:
    payload = {
      "observed": asdict(obs) | {"stock_terms": sorted(obs.stock_terms), "sp_terms": sorted(obs.sp_terms)},
      "watch": asdict(w) if w is not None else None,
      "rows": [{"num": r.row.num, "case": r.row.case, "venue": r.row.venue, "status": r.status,
                "reason": r.reason, "checks": [asdict(c) for c in r.checks]} for r in results],
    }
    Path(args.json).write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n  wrote {args.json}")

  failed = [r.row.num for r in results if r.status == FAIL]
  if failed:
    print(f"\nFAIL: rows {', '.join(str(n) for n in failed)} asserted state the device did not have.")
    return 1
  return 0


if __name__ == "__main__":
  sys.exit(main())
