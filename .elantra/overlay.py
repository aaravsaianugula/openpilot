#!/usr/bin/env python3
"""The overlay registry: every path by which this branch differs from upstream sunnypilot.

Single source of truth. sync.py restores from it, guards.py asserts against it,
verify_published.py checks the published branch against it, and
test_overlay_registration.py proves it still matches what is actually on the branch.

No imports on purpose -- everything else imports this, so it must never fail to import.

Adding a file to this branch means adding it here, in the same commit. A file that is not
here is deleted by the next sync, silently, because the rebuild starts from upstream and
puts back only what this list names.
"""

from __future__ import annotations

UPSTREAM_TINYGRAD = "sunnypilot/tinygrad"
FORK_TINYGRAD = "aaravsaianugula/tinygrad"

# Our NV-USB eGPU delta, replayed from tinygrad PR #17369 (russedavid:bounty/nv-usb3-wip).
#
# Invariant: exactly one non-merge commit on top of a commit that exists in
# sunnypilot/tinygrad. That makes diff-vs-merge-base exactly the patch and nothing else --
# which matters here in a way it does not for opendbc, because sunnypilot's tinygrad and
# upstream tinygrad are separate lineages. A merge-base computed across them can be months
# back, and the resulting "delta" would carry every upstream tinygrad change in between,
# silently moving the pin past the snapshot modeld is written against. That failure
# *succeeds*, which makes it far worse than a conflict.
TINYGRAD_PORT_BRANCH = "nv-usb3"
TINYGRAD_BRANCH = "nv-usb3-built"

# Overlay files that are entirely ours. Restored wholesale, so they cannot conflict.
OVERLAY_ADDED = [
    ".elantra",
    "CLAUDE.md",
    ".github/workflows/elantra-sync.yaml",
    ".github/workflows/elantra-guard.yaml",
    "openpilot/sunnypilot/egpu",
    "openpilot/selfdrive/ui/sunnypilot/mici/layouts/port_updates.py",
    "openpilot/selfdrive/ui/sunnypilot/mici/layouts/port_manifest.py",
    "openpilot/selfdrive/ui/sunnypilot/mici/layouts/egpu_state.py",
    "openpilot/selfdrive/ui/sunnypilot/mici/layouts/egpu_panel.py",
]

# Upstream files we modify. Kept deliberately tiny -- this is the only conflict surface in
# the whole design, so every line here has to earn its place. Prefer a new file in
# OVERLAY_ADDED plus a one-line import hook here over editing upstream code outright.
OVERLAY_MODIFIED = [
    ".gitmodules",
    "openpilot/common/params_keys.h",
    "openpilot/system/updated/updated.py",
    "openpilot/selfdrive/ui/sunnypilot/mici/layouts/settings.py",
    "openpilot/sunnypilot/modeld_v2/modeld.py",
    "openpilot/sunnypilot/models/manager.py",
    # Both carry the same one-line gate: upstream asks only whether a chestnut is attached,
    # never which card is behind it, so without these an eGPU we cannot drive fails the build
    # (blocking error window at boot) or takes the model path anyway. The logic lives in
    # openpilot/sunnypilot/egpu/guard.py, which is ours; these are the call sites.
    "openpilot/selfdrive/modeld/SConscript",
    "openpilot/selfdrive/modeld/modeld.py",
]

# Submodule pins we set ourselves. Not in OVERLAY_MODIFIED: they are index entries updated
# with `git update-index --cacheinfo`, not text the overlay patch can carry.
OVERLAY_GITLINKS = [
    "opendbc_repo",
    "tinygrad_repo",
]

# One or more hooks per file we modify, in that file's own vocabulary. These are the
# smallest strings that mean "our edit is still here" -- identifiers and key names, never a
# formatted source line, because upstream reflows code constantly and a whitespace-exact
# match is a check that fails for the wrong reason.
OVERLAY_HOOKS = {
    ".gitmodules": ["aaravsaianugula/opendbc", "aaravsaianugula/tinygrad"],
    "openpilot/common/params_keys.h": ["ElantraBuildManifest", "ElantraNewBuildManifest",
                                       "ElantraVerifiedUpdatesOnly", "EgpuVendor", "EgpuUseNvidia",
                                       "EgpuDeviceDetected"],
    "openpilot/system/updated/updated.py": ["get_elantra_manifest", "ElantraBuildManifest"],
    "openpilot/selfdrive/ui/sunnypilot/mici/layouts/settings.py": ["ElantraPortLayoutMici", "port_btn"],
    "openpilot/sunnypilot/modeld_v2/modeld.py": ["sunnypilot.egpu", "egpu.enabled",
                                                 "assert_pkl_matches", "guard.loading",
                                                 "small_model"],
    "openpilot/sunnypilot/models/manager.py": ["uses_amd_catalog", "probe_once"],
    "openpilot/selfdrive/modeld/SConscript": ["egpu_build_ok"],
    # smu.metrics() is the third hook because the inline read it replaced used SMU 13 field
    # names, which an SMU 11 card does not have; a sync that reverted it would put a 6600 XT
    # back to publishing no chestnutState at all, silently.
    "openpilot/selfdrive/modeld/modeld.py": ["sunnypilot.egpu", "egpu.enabled", "smu.metrics()"],
}

# The NV-USB delta may touch these and only these. Restricting the generated patch to them
# means it cannot carry unrelated churn; asserting the *unrestricted* diff stays inside them
# means a branch cut from the wrong base fails loudly instead of being quietly truncated.
NV_DELTA_PATHS = [
    "tinygrad/runtime/ops_nv.py",
    "tinygrad/runtime/support/nv/ip.py",
    "tinygrad/runtime/support/nv/nvdev.py",
    "tinygrad/runtime/support/system.py",
    "test/external/external_test_usb_asm24.py",
]

# Identifiers the patch introduces that stock tinygrad does not define. Names, not source
# lines: upstream reformats freely. Verified absent at sunnypilot/tinygrad 66ee3cfb.
NV_SENTINELS = {
    "tinygrad/runtime/ops_nv.py": [("class", "USBIface")],
    "tinygrad/runtime/support/nv/nvdev.py": [("class", "NVUSBPCIDevice")],
    "tinygrad/runtime/support/nv/ip.py": [("def", "shutdown_booter")],
}

# Files in NV_DELTA_PATHS with no sentinel of their own, and why. Without this, adding a
# path to the allowlist would quietly widen it with nothing checking the new file.
NV_SENTINEL_EXEMPT = {
    "tinygrad/runtime/support/system.py": "small edits to existing functions; no new definitions",
    "test/external/external_test_usb_asm24.py": "a one-line device-selection change in a test",
}

# "The class exists" and "anything ever constructs it" are different claims. Without the
# registry entry the USB backend is defined and nothing will ever select it.
NV_IFACE_REGISTRY = ("tinygrad/runtime/ops_nv.py", "NVDevice", "ifaces", "USBIface")
# The allocator's USB path branches on this. A botched three-way apply can reindent a method
# out of its class and the file still compiles, so assert it is a method of NVDevice.
NV_DEVICE_METHOD = ("tinygrad/runtime/ops_nv.py", "NVDevice", "is_usb")
