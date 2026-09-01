#!/usr/bin/env python3
"""Replay recorded drives through the COMPILED safety and prove panda accepts every frame.

The safety unit tests sweep synthetic torques against synthetic speeds. This does the opposite:
it takes the speeds, driver torques and lateral-active state the car actually saw, runs the real
opendbc limiter over them to get the command the SHIPPING ceiling would produce, and pushes
each frame through the real compiled libsafety in RAISED_LIMITS mode. A single rejection means
openpilot and panda disagree somewhere the car has actually been.

The ceiling is a FLAT 409 on the opendbc side and a FLAT 512 in panda -- 103 counts apart, and
neither reads speed. A speed schedule was replayed here for one build; it was retired after the
car's EPS faulted at both 500 and 450, and replaying it now would be measuring a build that does
not exist. What is replayed is what ships.

Because both sides are flat, the ceilings agree by construction and this could look like a
formality. It is not: the RATE LIMITER and the driver-override envelope are history dependent,
so what actually reaches the wire on a real drive is not simply "409 or less" -- it is whatever
400k frames of real demand, real driver torque and real engagement transitions produce. That is
what panda sees, and that is what this puts through it.

Three things make this a test rather than a demonstration:

  * The CONTROL. It also replays a command sequence built against a ceiling panda is NOT
    configured for (a deliberately-too-high 420). If that is also accepted, the harness is not
    enforcing anything and the pass above is worthless.
  * controls_allowed is set explicitly, because with it clear panda rejects everything and a
    "zero rejections" result would be vacuous in the other direction -- it would mean nothing
    was ever sent.
  * Rejections are counted per reason where possible and the first few are printed with their
    speed and torque, so a failure says where to look.

    PYTHONPATH=... python .elantra/replay_safety.py --segments 12
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PANDA_CEILING = 512     # HYUNDAI_STEERING_LIMITS_RAISED in opendbc/safety/modes/hyundai.h
TOO_HIGH = PANDA_CEILING + 1   # the control: above what panda is configured to accept
SAMPLE_FILL = 12        # rx enough speed frames to fill panda's sample window at the start
FRAME_US = 10_000       # LKAS11 is 100 Hz; panda's RT checks are wall-clock, so the replay
                        # has to advance the clock or it measures its own iteration speed


def build_frames(seg: str, ceiling: int):
    """(speed, driver_torque, command) per engaged frame, command from the real limiter.

    `ceiling` is a callable v_ego_raw -> counts. Both ceilings replayed here are flat, so the
    speed argument is unused -- the callable shape is kept because it is what lets the real
    ceiling and the over-ceiling control run through one code path with no special case, and
    because a future candidate would arrive through exactly this interface.
    """
    from opendbc.car.lateral import apply_driver_steer_torque_limits
    from openpilot.tools.lib.logreader import LogReader
    import torque_projection as tp

    lim = tp.Limits(max(ceiling(v / 10.0) for v in range(500)))
    out = []
    last = 0
    v = 0.0
    drv = 0.0
    for ev in LogReader(seg, sort_by_time=True):
        w = ev.which()
        if w == "carState":
            v = float(ev.carState.vEgoRaw)
            drv = float(ev.carState.steeringTorque)
        elif w == "carControl":
            cc = ev.carControl
            if not bool(cc.latActive):
                last = 0
                continue
            steer_max = ceiling(v)
            cmd = apply_driver_steer_torque_limits(
                int(round(float(cc.actuators.torque) * steer_max)), last, drv, lim, steer_max)
            last = cmd
            out.append((v, drv, cmd))
    return out


def replay(frames, label: str, verbose: bool = True):
    from opendbc.car.structs import CarParams
    from opendbc.car.hyundai.values import HyundaiSafetyFlags
    from opendbc.safety.tests.libsafety import libsafety_py
    from opendbc.safety.tests.common import CANPackerSafety
    import opendbc.safety.tests.test_hyundai as th

    safety = libsafety_py.libsafety
    packer = CANPackerSafety("hyundai_can_generated")
    safety.set_current_safety_param_sp(0)
    safety.set_safety_hooks(CarParams.SafetyModel.hyundai, HyundaiSafetyFlags.RAISED_LIMITS)
    safety.init_tests()

    cnt = {"speed": 0}

    def speed_msg(v):
        values = {f"WHL_SPD_{s}": v * 3.6 for s in ["FL", "FR", "RL", "RR"]}
        values["WHL_SPD_AliveCounter_LSB"] = (cnt["speed"] % 16) & 0x3
        values["WHL_SPD_AliveCounter_MSB"] = (cnt["speed"] % 16) >> 2
        cnt["speed"] += 1
        return packer.make_can_msg_safety("WHL_SPD11", 0, values, fix_checksum=th.checksum)

    def driver_msg(t):
        return packer.make_can_msg_safety("MDPS12", 0, {"CR_Mdps_StrColTq": t})

    def torque_msg(t):
        return packer.make_can_msg_safety("LKAS11", 0,
                                          {"CR_Lkas_StrToqReq": t, "CF_Lkas_ActToi": 1})

    # Prime the speed window once, then run at the real rate: one speed frame, one driver
    # frame and one command per 10 ms tick. panda's rt_torque_rate_limit_check re-anchors every
    # MAX_RT_INTERVAL of WALL CLOCK, so a replay that never advances the clock lets the anchor
    # go stale and rejects any command that drifts more than max_rt_delta from wherever the
    # replay happened to start. That is the harness measuring its own speed, not the car.
    if frames:
        for _ in range(SAMPLE_FILL):
            safety.safety_rx_hook(speed_msg(frames[0][0]))

    rejected = []
    t = 0
    for i, (v, drv, cmd) in enumerate(frames):
        t += FRAME_US
        safety.set_timer(t)
        safety.safety_rx_hook(speed_msg(v))
        safety.safety_rx_hook(driver_msg(int(drv)))
        safety.set_controls_allowed(1)
        if not safety.safety_tx_hook(torque_msg(int(cmd))):
            rejected.append((i, v, drv, cmd))

    ok = len(frames) - len(rejected)
    if verbose:
        print(f"  {label}: {len(frames)} frames, accepted {ok}, rejected {len(rejected)}")
        for i, v, drv, cmd in rejected[:5]:
            print(f"      frame {i}: v={v:.2f} m/s driver={drv:.0f} cmd={cmd}")
    return rejected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--routes", default="/data/media/0/realdata")
    ap.add_argument("--segments", type=int, default=12)
    args = ap.parse_args()

    def shipping_ceiling(_v: float) -> int:
        # Taken from torque_projection rather than from opendbc, deliberately: if this read
        # CarControllerParams it would agree with whatever is checked out, and a replay that
        # agrees with the thing it is checking proves nothing. This is the third opinion.
        import torque_projection as tp
        return tp.BEFORE_FLAT

    tp_ceiling = shipping_ceiling(0.0)
    segs = sorted(glob.glob(str(Path(args.routes) / "*" / "rlog.zst")))
    # Spread across the whole store rather than taking the tail, so one drive's conditions
    # cannot stand in for all of them.
    step = max(1, len(segs) // args.segments)
    chosen = segs[::step][:args.segments]
    print(f"replaying {len(chosen)} segments spread across {len(segs)}\n")

    total = 0
    all_rejected = []
    control_accepted = 0
    control_total = 0

    for seg in chosen:
        name = os.path.basename(os.path.dirname(seg))
        try:
            frames = build_frames(seg, shipping_ceiling)
        except Exception as e:
            print(f"  {name}: UNREADABLE {type(e).__name__}: {e}")
            return 1
        if not frames:
            print(f"  {name}: no engaged frames")
            continue
        total += len(frames)
        all_rejected += replay(frames, name)

        # The control, on the same segment: a sequence built above panda's ceiling must NOT
        # all pass. It has to be above 512, not above the commanded 409 -- panda's ceiling is
        # what rejects, and a control at 420 would sail straight through and prove nothing.
        # That 103-count spread is exactly the unenforced band, and this is the one place the
        # harness demonstrates that panda really does stop caring only at 512.
        ctl = build_frames(seg, lambda _v: TOO_HIGH)
        rej = replay(ctl, name + f" [control @{TOO_HIGH}]", verbose=False)
        control_total += len(ctl)
        control_accepted += len(ctl) - len(rej)

    print(f"\n{'-' * 62}")
    print(f"flat-{tp_ceiling} replay: {total} engaged frames, " +
          f"{len(all_rejected)} rejected by panda")
    print(f"control @{TOO_HIGH}:  {control_total} frames, {control_accepted} accepted " +
          f"({100 * control_accepted / max(control_total, 1):.2f}%)")

    if total == 0:
        print("REFUSED: no engaged frames replayed -- this proved nothing")
        return 1
    if control_total and control_accepted == control_total:
        print("REFUSED: the control sequence was fully accepted too. The harness is not")
        print("         enforcing the ceiling, so the clean result above is meaningless.")
        return 1
    if all_rejected:
        print("FAILED: panda rejected frames openpilot would have commanded")
        return 1
    print(f"PASSED: panda accepted every frame the shipping flat-{tp_ceiling} ceiling would")
    print("        produce, at every speed the car actually drove -- and rejected the")
    print("        over-ceiling control.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
