#!/usr/bin/env python3
"""Installed-package startup: isolated publisher selected without camera imports in card.

Runs only local Python processes. File reads/toggle/worker launch are replaced in
the bootstrap test; the actual bootstrap, authority provider and bridge are used.
No device connection, config installation, service operation or CAN transmission.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent

BOOTSTRAP = r'''
import io, json, os, sys, tempfile, uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from vision_long import native_bootstrap as bootstrap
from vision_long import observer_supervisor as supervisor

private = Path(sys.argv[1]).resolve()
assert Path(bootstrap.__file__).resolve().parent == private / "vision_long"
assert Path(supervisor.__file__).resolve().parent == private / "vision_long"
captured = []
class Toggle:
    def __init__(self, path): self.path = path
    def start(self): pass
    def stop(self): pass
    def enabled(self): return False
class Worker:
    is_alive, summary, failures, restarts = False, "fixture", 0, 0
    def __init__(self, bridge, **kwargs): captured.append((bridge, kwargs))
    def stop(self): pass
original_open = Path.open
def read(path, *args, **kwargs):
    if path == Path("/data/vision_long/fixture-key"):
        return io.BytesIO(b"test-key-not-a-secret-01234567890123456789")
    return original_open(path, *args, **kwargs)
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    boot = str(uuid.uuid4())
    (root / "boot").write_text(boot)
    config = root / "config.json"
    config.write_text(json.dumps(dict(schema="native_runtime_v2", listen_ip="127.0.0.1", port=45556,
                                     key_file="/data/vision_long/fixture-key", toggle_file="/data/vision_long/braking")))
    controller = SimpleNamespace(vision_long_bridge=None, vision_long_consumer=None)
    with patch.object(bootstrap, "BrakingToggle", Toggle), patch.object(bootstrap, "BOOT_ID_FILE", root / "boot"), \
         patch.object(bootstrap, "boottime_ns", lambda: 1000000000), patch.object(Path, "open", read), \
         patch.object(supervisor, "IsolatedPublisher", Worker):
        binding = bootstrap.start_native_runtime(controller, config)
assert len(captured) == 1
bridge, options = captured[0]
assert controller.vision_long_bridge is binding and binding.bridge is bridge
assert options["command"] == [sys.executable, "-m", "vision_long_publisher.comma_observer"]
assert options["boot_id"] == boot
assert Path(options["cwd"]).resolve() == private
assert Path(options["env"]["PYTHONPATH"].split(os.pathsep)[0]).resolve() == private
assert bridge.trusted_clock_id == boot + ":CLOCK_BOOTTIME"
assert not bridge.braking_enabled()
assert "vision_long_publisher.comma_context_publisher" not in sys.modules
assert "vision_long.camera_io" not in sys.modules
assert binding.stage_cycle(1000000000) is None
print("installed isolated bootstrap/import contract passed")
'''


class InstalledObserverTests(unittest.TestCase):
  def environment(self):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(HERE)
    return environment

  def test_bootstrap_selects_isolated_process_without_camera_imports(self):
    result = subprocess.run([sys.executable, "-c", BOOTSTRAP, str(HERE)], cwd=HERE.parent, env=self.environment(), capture_output=True, text=True, timeout=10)
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("installed isolated bootstrap/import contract passed", result.stdout)

  def test_private_observer_entry_resolves_as_a_module(self):
    result = subprocess.run(
      [sys.executable, "-m", "vision_long_publisher.comma_observer", "--help"],
      cwd=HERE.parent,
      env=self.environment(),
      capture_output=True,
      text=True,
      timeout=10,
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("--incarnation", result.stdout)
    self.assertIn("--boot-id", result.stdout)


if __name__ == "__main__":
  unittest.main()
