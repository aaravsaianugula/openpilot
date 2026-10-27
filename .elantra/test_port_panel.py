#!/usr/bin/env python3
"""
Tests for the Elantra port panel's decision logic.

The panel needs pyray and a running UI, which CI cannot stand up. Its *decisions* --
when Install is offered, when it is held back, how a missing or malformed manifest is
treated -- are pure, and live in port_manifest.py precisely so they can be tested. This
imports that module by path, so it exercises the code that ships rather than a copy of
it that would quietly drift.

The case worth being careful about: a staged build with no manifest is not one of our
builds, because the user deliberately switched branches. Blocking it would strand them,
so the gate does not apply. That is a decision, not an oversight, and it is pinned here
so nobody "fixes" it later.

    python .elantra/test_port_panel.py
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import sys
from pathlib import Path

MODULE = (Path(__file__).resolve().parent.parent
          / "openpilot/selfdrive/ui/sunnypilot/mici/layouts/port_manifest.py")


def load_module():
    if not MODULE.is_file():
        raise SystemExit(
            f"port_manifest.py is missing at {MODULE}.\n"
            + "The panel's decision logic is gone, which means the update gate is gone.")
    spec = importlib.util.spec_from_file_location("port_manifest", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pm = load_module()

GREEN = json.dumps({
    "sunnypilot_upstream_sha": "b742b96c4482aced861525c33c55e1374aa8bb0c",
    "sunnypilot_upstream_date": "2026-08-25",
    "upstream_ci_conclusion": "success",
    "upstream_ci_checked": 21,
    "opendbc_sha": "69e2e548f" + "0" * 31,
    "elantra_platforms": ["HYUNDAI_ELANTRA_2024", "HYUNDAI_ELANTRA_HEV_2024"],
    "synced_at_utc": "2026-08-25T21:00:00Z",
})
RED = json.dumps({**json.loads(GREEN), "upstream_ci_conclusion": "failure"})
RUNNING = json.dumps({**json.loads(GREEN), "upstream_ci_conclusion": "in_progress"})

failures: list[str] = []


def case(name: str, got, want) -> None:
    ok = got == want
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        failures.append(name)


def main() -> int:
    print(f"Elantra port panel logic\n  module: {MODULE}\n")

    print("[install gate] verified builds only ON")
    case("green build is offered", pm.install_offered(True, True, GREEN), True)
    case("red build is held back", pm.install_offered(True, True, RED), False)
    case("in-progress build is held back", pm.install_offered(True, True, RUNNING), False)
    case("held-back build shows the explanation", pm.update_blocked(True, True, RED), True)
    case("green build shows no explanation", pm.update_blocked(True, True, GREEN), False)

    print("\n[install gate] verified builds only OFF")
    case("red build is offered", pm.install_offered(True, False, RED), True)
    case("nothing is ever held back", pm.update_blocked(True, False, RED), False)

    print("\n[no update pending]")
    case("no install button", pm.install_offered(False, True, GREEN), False)
    case("no explanation either", pm.update_blocked(False, True, RED), False)

    print("\n[not one of our builds] -- deliberate: do not strand a deliberate branch switch")
    case("absent manifest is not blocked", pm.install_offered(True, True, ""), True)
    case("absent manifest shows no block note", pm.update_blocked(True, True, ""), False)
    case("None manifest is not blocked", pm.install_offered(True, True, None), True)

    print("\n[malformed input] -- must not raise, must not block")
    case("truncated json", pm.install_offered(True, True, '{"upstream_ci'), True)
    case("json array not object", pm.install_offered(True, True, "[1,2,3]"), True)
    case("bare string", pm.install_offered(True, True, "not json at all"), True)
    case("bytes are decoded", pm.install_offered(True, True, GREEN.encode()), True)
    case("bytes, red, still blocked", pm.install_offered(True, True, RED.encode()), False)

    print("\n[manifest parsing]")
    m = pm.load_manifest(GREEN)
    case("upstream sha round-trips", m["sunnypilot_upstream_sha"][:9], "b742b96c4")
    case("platform list round-trips", len(m["elantra_platforms"]), 2)
    case("garbage yields empty dict", pm.load_manifest("{{{"), {})
    case("green manifest reads as green", pm.ci_is_green(m), True)
    case("red manifest does not", pm.ci_is_green(pm.load_manifest(RED)), False)

    print("\n[last synced]")
    now = datetime.datetime(2026, 8, 30, 12, 0, tzinfo=datetime.UTC)
    case("five days", pm.age("2026-08-25T12:00:00Z", now), "5 days ago")
    case("weekly cadence reads as days", pm.age("2026-08-23T12:00:00Z", now), "7 days ago")
    case("three hours", pm.age("2026-08-30T09:00:00Z", now), "3 hours ago")
    case("minutes round to just now", pm.age("2026-08-30T11:30:00Z", now), "just now")
    case("missing timestamp", pm.age(None, now), "unknown")
    case("garbage timestamp", pm.age("not-a-date", now), "unknown")
    # Devices boot with a bad clock often enough that a future timestamp is a real case.
    case("clock skew does not produce nonsense",
         pm.age("2026-09-05T12:00:00Z", now), "just now")

    print("\n" + "-" * 58)
    if failures:
        print(f"FAILED: {len(failures)} case(s)")
        for f in failures:
            print("  - " + f)
        return 1
    print("PASSED: panel logic behaves as designed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
