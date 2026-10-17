#!/usr/bin/env python3
"""Tests for the Vision Long companion seam in the Hyundai carcontroller.

The companion is braking-only. Three properties make it safe to ship in the car process, and
each is tested here against the REAL CarController, the real CAN packer and the real consumer:

    * No runtime config -> the companion code never runs. Every CAN message and every reported
        actuator is byte-identical to the unpatched native controller at the reviewed base commit.
    * A malformed or unprovisioned config -> the bootstrap fails closed. Same byte-identical
        native output, and the failure is visible in vision_long_bootstrap_status.
    * With a consumer bound and caps arriving (including caps ABOVE native), SCC12 aReqRaw and
        aReqValue are never above native, StopReq and every other field stay exact, and a driver
        override gives exact native bytes.
    * The state is visible: one carlog line (card forwards carlog to cloudlog) at startup and on
        every change, "vision_long_companion mode=... authority=... reason=...", never per cycle.

The native reference is the controller and CAN builder at the commit recorded in
vision_long/overlay-receipt.json, read out of opendbc's own history and checked against the
receipt's hashes, so the comparison is against the code the car ran before this branch.

Not tested here: the diagnostic publisher worker (it needs cereal/msgq on the device) and any
live clock or physical qualification, which this package does not supply.

        python .elantra/test_vision_long_companion.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OPENDBC = ROOT / "opendbc_repo"
COMPANION = HERE / "vision_long"
RECEIPT = COMPANION / "overlay-receipt.json"
RUNTIME_CONFIG = COMPANION / "native-runtime.json"
CONTROLLER_PATH = "opendbc/car/hyundai/carcontroller.py"
HYUNDAICAN_PATH = "opendbc/car/hyundai/hyundaican.py"

sys.path.insert(0, str(OPENDBC))
sys.path.insert(0, str(HERE))

import opendbc

if not Path(opendbc.__file__).resolve().is_relative_to(OPENDBC):
    raise RuntimeError("opendbc resolves to " + opendbc.__file__ + ", not this tree's " + str(OPENDBC))

from opendbc.can import CANParser
from opendbc.car import DT_CTRL, Bus, gen_empty_fingerprint, structs
from opendbc.car.hyundai import carcontroller as patched_carcontroller
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR, DBC
from vision_long.integration import CameraReference, ControlGate, ProposalPacket, sign_packet
from vision_long.native_bootstrap import bind_native_controller
from vision_long.native_bridge import NativeReturnBridge
from vision_long.native_consumer import NativeCycle, NativeScc12Consumer

SCC12_ADDR = 0x421
FRAME_NS = int(DT_CTRL * 1e9)  # 100 Hz controller
PACKET_FRAMES = 5  # 20 Hz companion packets
RETURN_LATENCY_NS = 10_000_000  # capture to arrival, inside the consumer's return-age budget
SCENARIO_FRAMES = 600
NATIVE_ACCEL_MIN, NATIVE_ACCEL_MAX = -3.5, 2.0
# Caps reach above native's own maximum on purpose: a cap must never raise a field.
CAP_ACCEL_MIN, CAP_ACCEL_MAX = -3.5, 2.0
KEY = bytes(range(32))
SESSION = str(uuid.UUID(int=11))
BUILD = "vision-long-companion-test"
LONG_CONTROL_STATES = (
    structs.CarControl.Actuators.LongControlState.pid,
    structs.CarControl.Actuators.LongControlState.stopping,
    structs.CarControl.Actuators.LongControlState.starting,
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def receipt() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def base_source(relative: str) -> bytes:
    """The unpatched file at the receipt's opendbc commit, verified against the receipt hash."""
    record = receipt()
    commit = record["reference_opendbc_commit"]
    shown = subprocess.run(("git", "show", commit + ":" + relative), cwd=str(OPENDBC),
                                                  capture_output=True, check=False)
    if shown.returncode != 0:
        raise RuntimeError("opendbc history lacks the reviewed base " + commit + ": " + shown.stderr.decode())
    expected = record["base_files"]["opendbc_repo/" + relative]
    if sha256(shown.stdout) != expected:
        raise RuntimeError("base " + relative + " at " + commit + " does not match the receipt")
    return shown.stdout


def forget_companion_modules() -> None:
    for name in [name for name in sys.modules if name == "vision_long" or name.startswith("vision_long.")]:
        del sys.modules[name]


class Sandbox:
    """A copy of the patched controller under a throwaway root, so a runtime config can be
    placed next to it without ever writing into this tree. The controller resolves
    .elantra/vision_long relative to its own file, exactly as it does on the device."""

    def __init__(self, config: bytes | None):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        controller = root / "opendbc_repo" / CONTROLLER_PATH
        controller.parent.mkdir(parents=True)
        shutil.copyfile(OPENDBC / CONTROLLER_PATH, controller)
        shutil.copytree(COMPANION, root / ".elantra/vision_long")
        if config is not None:
            (root / ".elantra/vision_long/native-runtime.json").write_bytes(config)
        self.support = str(root / ".elantra")
        forget_companion_modules()
        self.module = load_module("vision_long_sandbox_carcontroller", controller)

    def close(self) -> None:
        forget_companion_modules()
        while self.support in sys.path:
            sys.path.remove(self.support)
        self.tmp.cleanup()


def native_reference() -> type:
    """The base CarController wired to the base CAN builder, both from opendbc history."""
    holder = tempfile.TemporaryDirectory()
    folder = Path(holder.name)
    (folder / "hyundaican.py").write_bytes(base_source(HYUNDAICAN_PATH))
    (folder / "carcontroller.py").write_bytes(base_source(CONTROLLER_PATH))
    base_can = load_module("vision_long_reference_hyundaican", folder / "hyundaican.py")
    base_controller = load_module("vision_long_reference_carcontroller", folder / "carcontroller.py")
    base_controller.hyundaican = base_can  # every call site is a module attribute lookup
    holder.cleanup()
    return base_controller.CarController


def interface_with(controller_class: type) -> CarInterface:
    fingerprint = gen_empty_fingerprint()
    CP = CarInterface.get_params(CAR.HYUNDAI_ELANTRA_2024, fingerprint, [], alpha_long=True,
                                                              is_release=False, docs=False)
    CP_SP = CarInterface.get_params_sp(CP, CAR.HYUNDAI_ELANTRA_2024, fingerprint, [], alpha_long=True,
                                                                          is_release_sp=False, docs=False)
    if not CP.openpilotLongitudinalControl:
        raise RuntimeError("the companion seam needs openpilot longitudinal on the Elantra")
    interface = CarInterface(CP, CP_SP)
    dbc_names = {bus: parser.dbc_name for bus, parser in interface.can_parsers.items()}
    interface.CC = controller_class(dbc_names, CP, CP_SP)
    return interface


def scenario(seed: int) -> list[dict]:
    """A seeded drive: accel across native's range, stops, overrides, pedals, standstill."""
    rng = random.Random(seed)
    frames = []
    for _ in range(SCENARIO_FRAMES):
        frames.append({
            "enabled": rng.random() > 0.05,
            "long_active": rng.random() > 0.05,
            "accel": round(rng.uniform(NATIVE_ACCEL_MIN, NATIVE_ACCEL_MAX), 3),
            "long_state": rng.choice(LONG_CONTROL_STATES),
            "override": rng.random() < 0.08,
            "brake": rng.random() < 0.05,
            "gas": rng.random() < 0.05,
            "standstill": rng.random() < 0.10,
        })
    return frames


def step(interface: CarInterface, frame: dict, now_ns: int):
    interface.update([])
    interface.CS.out.brakePressed = frame["brake"]
    interface.CS.out.gasPressed = frame["gas"]
    interface.CS.out.standstill = frame["standstill"]
    CC = structs.CarControl()
    CC.enabled = frame["enabled"]
    CC.longActive = frame["long_active"]
    CC.actuators.accel = frame["accel"]
    CC.actuators.longControlState = frame["long_state"]
    CC.cruiseControl.override = frame["override"]
    return interface.apply(CC.as_reader(), structs.CarControlSP(), now_ns)


def snapshot(result) -> tuple[dict, list[tuple]]:
    """Comparable form of one apply(): the actuator fields and every CAN message."""
    actuators, sends = result
    return actuators.to_dict(), [tuple(message) for message in sends]


def scc12(sends) -> list:
    return [message for message in sends if message[0] == SCC12_ADDR]


class Scc12Decoder:
    def __init__(self):
        self.parser = CANParser(DBC[CAR.HYUNDAI_ELANTRA_2024][Bus.pt], [("SCC12", 0)], 0)
        self.now_ns = 0

    def decode(self, message) -> dict:
        self.now_ns += 2 * FRAME_NS
        self.parser.update([(self.now_ns, [message])])
        return dict(self.parser.vl["SCC12"])


class DefaultIsNativeTests(unittest.TestCase):
    """No config file -> the companion never runs; the car gets exactly the base controller."""

    def test_the_shipped_tree_carries_no_runtime_config(self):
        self.assertFalse(RUNTIME_CONFIG.exists(),
                                          "a runtime config in the delivery tree would start the companion on every boot")

    def test_the_installed_files_are_exactly_the_reviewed_generated_overlay(self):
        outputs = receipt()["output_files"]
        for relative in ("opendbc_repo/" + CONTROLLER_PATH, "opendbc_repo/" + HYUNDAICAN_PATH):
            with self.subTest(relative=relative):
                self.assertEqual(sha256((ROOT / relative).read_bytes()), outputs[relative])
        for path in sorted(COMPANION.glob("*.py")) + sorted((HERE / "vision_long_publisher").glob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            with self.subTest(relative=relative):
                self.assertEqual(sha256(path.read_bytes()), outputs[relative])

    def test_absent_config_disables_the_companion_and_output_is_byte_identical_to_native(self):
        patched = interface_with(patched_carcontroller.CarController)
        reference = interface_with(native_reference())
        self.assertEqual(patched.CC.vision_long_bootstrap_status, "disabled_no_runtime_config")
        self.assertIsNone(patched.CC.vision_long_bridge)
        now = 0
        for index, frame in enumerate(scenario(seed=1)):
            expected = snapshot(step(reference, frame, now))
            actual = snapshot(step(patched, frame, now))
            self.assertIsNone(patched.CC.vision_long_consumer, index)
            self.assertIsNone(patched.CC.vision_long_boundary, index)
            self.assertEqual(actual[1], expected[1], ("CAN differs from native", index, frame))
            self.assertEqual(actual[0], expected[0], ("actuators differ from native", index, frame))
            now += FRAME_NS


class MalformedConfigTests(unittest.TestCase):
    """A config that is present but unusable fails closed to native, visibly."""

    VALID_SHAPE = {
        "schema": "native_diagnostic_return_v1",
        "listen_ip": "192.168.5.1",
        "port": 47100,
        "key_file": "/data/vision-long-test-key-that-does-not-exist",
        "pairs": 10,
        "max_duration_s": 60,
    }
    CASES = {
        "not json": (b"{authority: true", "native_bootstrap_failed:JSONDecodeError"),
        "empty": (b"", "native_bootstrap_failed:ValueError"),
        "authority field": (json.dumps({**VALID_SHAPE, "authority": True}).encode(), "native_bootstrap_failed:ValueError"),
        "wrong schema": (json.dumps({**VALID_SHAPE, "schema": "native_authority_v1"}).encode(),
                                          "native_bootstrap_failed:ValueError"),
        "key outside /data": (json.dumps({**VALID_SHAPE, "key_file": "/tmp/key"}).encode(),
                                                    "native_bootstrap_failed:ValueError"),
        "ipv6": (json.dumps({**VALID_SHAPE, "listen_ip": "::1"}).encode(), "native_bootstrap_failed:ValueError"),
        "unprovisioned key": (json.dumps(VALID_SHAPE).encode(), "native_bootstrap_failed:FileNotFoundError"),
    }

    def test_every_malformed_config_fails_closed_to_byte_identical_native(self):
        reference_class = native_reference()
        frames = scenario(seed=2)[:120]
        for name, (config, status) in self.CASES.items():
            with self.subTest(case=name):
                sandbox = Sandbox(config)
                try:
                    patched = interface_with(sandbox.module.CarController)
                    reference = interface_with(reference_class)
                    self.assertEqual(patched.CC.vision_long_bootstrap_status, status)
                    self.assertIsNone(patched.CC.vision_long_bridge)
                    now = 0
                    for index, frame in enumerate(frames):
                        expected = snapshot(step(reference, frame, now))
                        actual = snapshot(step(patched, frame, now))
                        self.assertIsNone(patched.CC.vision_long_boundary, index)
                        self.assertEqual(actual, expected, (name, index))
                        now += FRAME_NS
                finally:
                    sandbox.close()

    def test_the_sandbox_controller_is_the_shipped_controller(self):
        sandbox = Sandbox(None)
        try:
            self.assertEqual(Path(sandbox.module.__file__).read_bytes(), (OPENDBC / CONTROLLER_PATH).read_bytes())
            self.assertEqual(interface_with(sandbox.module.CarController).CC.vision_long_bootstrap_status,
                                              "disabled_no_runtime_config")
        finally:
            sandbox.close()


class BrakingOnlyTests(unittest.TestCase):
    """A bound consumer fed caps across the whole range never raises a native SCC12 field."""

    def drive(self, seed: int):
        """Lockstep patched and reference controllers; returns (frame, native, companion, eligible)."""
        rng = random.Random(seed)
        gate = ControlGate(KEY, SESSION, BUILD, CAP_ACCEL_MIN, CAP_ACCEL_MAX)
        clock = {"now": 0}
        consumer = NativeScc12Consumer(gate, completion_clock=lambda: clock["now"])
        patched = interface_with(patched_carcontroller.CarController)
        patched.CC.vision_long_consumer = consumer  # the trusted bridge's job, done by the test
        reference = interface_with(native_reference())
        decoder_native, decoder_companion = Scc12Decoder(), Scc12Decoder()
        rows = []
        sequence = 0
        pending: list[tuple[int, bytes]] = []
        last_sent: float | None = None
        base = 3_000_000_000
        for index, frame in enumerate(scenario(seed)):
            now = base + index * FRAME_NS
            clock["now"] = now
            if index % PACKET_FRAMES == 0:
                sequence += 1
                cameras = (CameraReference("narrow", sequence, now - 1_000_000), CameraReference("wide", sequence, now))
                gate.issue(sequence, cameras, now)
                cap = round(rng.uniform(CAP_ACCEL_MIN, CAP_ACCEL_MAX), 3)
                packet = ProposalPacket(SESSION, BUILD, sequence, sequence, cameras, cap, False, 0.99, True, ())
                pending.append((now + RETURN_LATENCY_NS, sign_packet(packet, KEY)))
            arrived = None
            if pending and pending[0][0] <= now:
                arrived = pending.pop(0)[1]
            consumer.stage(NativeCycle(now, arrived))
            expected_actuators, expected = step(reference, frame, now)
            actual_actuators, actual = step(patched, frame, now)
            boundary = patched.CC.vision_long_boundary
            eligible = boundary is not None and boundary.decision is not None and boundary.decision.companion_eligible
            native_messages, companion_messages = scc12(expected), scc12(actual)
            self.assertEqual(len(native_messages), len(companion_messages), index)
            non_scc12 = [message for message in actual if message[0] != SCC12_ADDR]
            self.assertEqual(non_scc12, [message for message in expected if message[0] != SCC12_ADDR], index)
            if native_messages:
                # The accel reported on an SCC12 frame is what was sent: never above native.
                self.assertLessEqual(actual_actuators.accel, expected_actuators.accel + 1e-6, index)
                last_sent = actual_actuators.accel if eligible else None
            elif actual_actuators.accel != expected_actuators.accel:
                # Between SCC12 frames the wire still carries the last companion SCC12, and
                # that is what gets reported, rather than a native value that was never sent.
                self.assertEqual(actual_actuators.accel, last_sent, index)
            for native_message, companion_message in zip(native_messages, companion_messages, strict=True):
                rows.append((frame, native_message, companion_message, eligible,
                                          decoder_native.decode(native_message), decoder_companion.decode(companion_message)))
        return rows

    def test_no_scc12_field_is_ever_raised_and_stop_request_is_exact(self):
        lowered = 0
        for seed in (3, 4, 5):
            for frame, native_message, companion_message, eligible, native, companion in self.drive(seed):
                with self.subTest(seed=seed, frame=frame):
                    self.assertLessEqual(companion["aReqRaw"], native["aReqRaw"] + 1e-6)
                    self.assertLessEqual(companion["aReqValue"], native["aReqValue"] + 1e-6)
                    self.assertEqual(companion["StopReq"], native["StopReq"])
                    for signal in native:
                        if signal not in ("aReqRaw", "aReqValue", "CR_VSM_ChkSum"):
                            self.assertEqual(companion[signal], native[signal], signal)
                    if not eligible:
                        self.assertEqual(companion_message, native_message)
                lowered += companion["aReqValue"] < native["aReqValue"] - 1e-6
        # The property must not hold vacuously: the companion has to have actually braked.
        self.assertGreater(lowered, 0)

    def test_driver_override_and_pedals_give_exact_native_bytes(self):
        for frame, native_message, companion_message, eligible, _, _ in self.drive(seed=6):
            if frame["override"] or frame["brake"] or frame["gas"]:
                self.assertFalse(eligible, frame)
                self.assertEqual(companion_message, native_message, frame)


STATUS_PREFIX = "vision_long_companion "
CAMERA_CLOCK = "status-test-boot:CLOCK_BOOTTIME"


class StatusCapture(logging.Handler):
    """Every companion status line carlog emits while active, with its level."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines: list[tuple[int, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith(STATUS_PREFIX):
            self.lines.append((record.levelno, message))

    def __enter__(self) -> StatusCapture:
        logging.getLogger("carlog").addHandler(self)
        return self

    def __exit__(self, *exc_info) -> None:
        logging.getLogger("carlog").removeHandler(self)

    def messages(self) -> list[str]:
        return [message for _, message in self.lines]


def status_line(mode: str, authority: str, reason: str) -> str:
    return f"vision_long_companion mode={mode} authority={authority} reason={reason}"


def started_with_capture(controller_class: type) -> tuple[CarInterface, StatusCapture]:
    """interface_with builds two controllers (the interface's own, then the one under test), so
    capture exactly one construction: the controller card would run, as card builds it once."""
    interface = interface_with(controller_class)
    dbc_names = {bus: parser.dbc_name for bus, parser in interface.can_parsers.items()}
    with StatusCapture() as capture:
        interface.CC = controller_class(dbc_names, interface.CP, interface.CP_SP)
    return interface, capture


class StatusLogTests(unittest.TestCase):
    """The companion's state is logged once at startup and on each change, never per cycle."""

    def test_absent_config_logs_disabled_once_at_startup_and_never_per_cycle(self):
        patched, capture = started_with_capture(patched_carcontroller.CarController)
        self.assertEqual(capture.messages(), [status_line("diagnostic_only", "disabled", "disabled_no_runtime_config")])
        with capture:
            now = 0
            for frame in scenario(seed=7)[:200]:
                step(patched, frame, now)
                now += FRAME_NS
        self.assertEqual(len(capture.lines), 1)
        # card's ForwardingHandler hands it to cloudlog; carlog's own level (INFO) lets it through.
        self.assertGreaterEqual(capture.lines[0][0], logging.getLogger("carlog").getEffectiveLevel())

    def test_malformed_config_logs_the_fail_closed_reason(self):
        sandbox = Sandbox(b"{authority: true")
        try:
            _, capture = started_with_capture(sandbox.module.CarController)
            self.assertEqual(capture.messages(), [status_line(
                "diagnostic_only", "disabled", "native_bootstrap_failed:JSONDecodeError")])
        finally:
            sandbox.close()

    def test_a_bound_session_logs_enabled_while_in_the_path_and_disabled_when_it_closes(self):
        gate = ControlGate(KEY, SESSION, BUILD, CAP_ACCEL_MIN, CAP_ACCEL_MAX)
        clock = {"now": 0}
        bridge = NativeReturnBridge(lambda session: gate, completion_clock=lambda: clock["now"],
                                                                trusted_clock_id=CAMERA_CLOCK)
        patched = interface_with(patched_carcontroller.CarController)
        frames = scenario(seed=8)
        now = 3_000_000_000

        def run(count: int) -> None:
            nonlocal now
            for _ in range(count):
                clock["now"] = now
                step(patched, frames[(now // FRAME_NS) % len(frames)], now)
                now += FRAME_NS

        with StatusCapture() as capture:
            bind_native_controller(patched.CC, bridge)
            run(5)
            bridge.open_session(SESSION, CAMERA_CLOCK)
            run(20)
            self.assertIsNotNone(patched.CC.vision_long_consumer)
            bridge.close_session(SESSION)
            run(20)
            self.assertIsNone(patched.CC.vision_long_consumer)
        self.assertEqual(capture.messages(), [
            status_line("diagnostic_only", "disabled", "awaiting_session"),
            status_line("braking_only", "enabled", "staged_without_invented_native_evidence"),
            status_line("diagnostic_only", "disabled", "native_session_closed"),
        ])


if __name__ == "__main__":
    unittest.main()
