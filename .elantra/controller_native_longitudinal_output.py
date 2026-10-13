"""Stateful native Hyundai software-report diagnostic, without CAN or a plant.

Each step is a conditional publication/loop event, not an assumed 100Hz apply.
Publication precedes optional application. Cold state is explicitly uncertain;
software acceleration reports do not establish safety acceptance or delivery.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
import inspect
import math
from numbers import Real
import os
from pathlib import Path
import re
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace

import controller_native_search as search

SUPERPROJECT_HEAD = "13aedc522518fe1b62651d9a8a320e931efc335c"
OPENDBC_PIN = "b45eb78def29582bcce43c68cfb1d55f19224d1a"
CONTROLLER = "opendbc_repo/opendbc/sunnypilot/car/hyundai/longitudinal/controller.py"
PINNED = {
  CONTROLLER: "0832a390bf48198105e7c0ea1276cdb9708ee4d83339158c4d149d58dbc434da",
  "opendbc_repo/opendbc/sunnypilot/car/hyundai/longitudinal/helpers.py": "f06d75d7a2bb36e78ab47df29305c6dd9ee775e314de3e313ffd1b0b8f0172a0",
  "opendbc_repo/opendbc/sunnypilot/car/hyundai/longitudinal/config.py": "4f79ef3c1039c0d6f67c38bee40970d73ddd2d805a1a40d4501e95f467cbb092",
  "opendbc_repo/opendbc/car/hyundai/carcontroller.py": "29fd6c14ee583b7f7a432481b549235ef56d2a58f08f28b3ed6b6fc8c7c79ac4",
  "openpilot/selfdrive/car/card.py": "6e54bc5145634e451d6b48de9dd80691bf891ba81fab77f7d4de59eebe6c191b",
  "opendbc_repo/opendbc/car/hyundai/values.py": "6df5d2c82c6d1b16ae108e30eec16136e34b20862d73186a685b4bc59504957c",
  "opendbc_repo/opendbc/sunnypilot/car/hyundai/values.py": "f1e36735bf9f166a84453627a70cb7668f2156850a7df991470470070590d4ec",
  "opendbc_repo/opendbc/car/__init__.py": "49d8cb964d26577791dff0d002be834b0de690a7168cab990043ed1a036ecf11",
  "opendbc_repo/opendbc/car/structs.py": "33b79349c95431b1bdfcc450200e3d74689bc1c5571d7aa6887c67ffa0c9283c",
  "opendbc_repo/opendbc/car/interfaces.py": "75f378c4dafc658e54f60ae374907bf966ee47c85afec02d9d008f996355112e",
  "opendbc_repo/opendbc/car/car.capnp": "b971083a0cb3aed287ca9c54572d470fb798e633bc47611717f697b9e735f5a1",
  "openpilot/cereal/custom.capnp": "d439fdd23e665ec2d74039619c903219386ac6ce3ddadfbc279ae669b9dd1cd6",
  "openpilot/cereal/__init__.py": "72fea22180ef49968dd924360ef0638b762680dd4f0931fbb8a8cebf69574e22",
}
SOURCES = tuple(PINNED)
# Parent initializers in the bounded top-level native import closure, plus the
# recorded card source parents. None denotes a namespace initializer's absence.
BOOTSTRAP = dict.fromkeys(
  (
    "openpilot/__init__.py",
    "openpilot/common/__init__.py",
    "openpilot/selfdrive/__init__.py",
    "openpilot/selfdrive/car/__init__.py",
    "opendbc_repo/opendbc/car/body/__init__.py",
    "opendbc_repo/opendbc/car/chrysler/__init__.py",
    "opendbc_repo/opendbc/car/common/__init__.py",
    "opendbc_repo/opendbc/car/ford/__init__.py",
    "opendbc_repo/opendbc/car/gm/__init__.py",
    "opendbc_repo/opendbc/car/honda/__init__.py",
    "opendbc_repo/opendbc/car/hyundai/__init__.py",
    "opendbc_repo/opendbc/car/mazda/__init__.py",
    "opendbc_repo/opendbc/car/mg/__init__.py",
    "opendbc_repo/opendbc/car/mock/__init__.py",
    "opendbc_repo/opendbc/car/nissan/__init__.py",
    "opendbc_repo/opendbc/car/psa/__init__.py",
    "opendbc_repo/opendbc/car/rivian/__init__.py",
    "opendbc_repo/opendbc/car/subaru/__init__.py",
    "opendbc_repo/opendbc/car/tesla/__init__.py",
    "opendbc_repo/opendbc/car/toyota/__init__.py",
    "opendbc_repo/opendbc/car/volkswagen/__init__.py",
    "opendbc_repo/opendbc/sunnypilot/__init__.py",
    "opendbc_repo/opendbc/sunnypilot/car/gm/__init__.py",
    "opendbc_repo/opendbc/sunnypilot/car/honda/__init__.py",
    "opendbc_repo/opendbc/sunnypilot/car/hyundai/__init__.py",
    "opendbc_repo/opendbc/sunnypilot/car/hyundai/longitudinal/__init__.py",
    "opendbc_repo/opendbc/sunnypilot/car/subaru/__init__.py",
    "opendbc_repo/opendbc/sunnypilot/car/tesla/__init__.py",
    "opendbc_repo/opendbc/sunnypilot/car/toyota/__init__.py",
  ),
  "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
)
BOOTSTRAP.update(
  {
    "msgq_repo/msgq/__init__.py": "9b9e5e26cf4190e9aaf056cd90771fb95228f665d1171a9ee919b432a0d2d368",
    "opendbc_repo/opendbc/__init__.py": "b0fad4b497d4f496ef68d75656228bd4db18e5774a21808a4a53bd2fdadecedf",
    "opendbc_repo/opendbc/can/__init__.py": "12b5637232dc8c199b131e169004b27825c021c2e1f741e391ba9f480f0d41a2",
    "opendbc_repo/opendbc/car/__init__.py": "49d8cb964d26577791dff0d002be834b0de690a7168cab990043ed1a036ecf11",
    "opendbc_repo/opendbc/sunnypilot/car/__init__.py": "05b0c1314ef5854b44794930c80b27065af31cdf4ceb5d50f6259453ec39c50d",
    "openpilot/cereal/__init__.py": "72fea22180ef49968dd924360ef0638b762680dd4f0931fbb8a8cebf69574e22",
    "openpilot/cereal/messaging/__init__.py": "ef101de88d76a55f5724c6142d7f073a21b9024eb5b2ffce49ca7b9e5ea929f0",
  }
)
TUNING_FIELDS = ("desired_accel", "actual_accel", "accel_last", "jerk_upper", "jerk_lower", "comfort_band_upper", "comfort_band_lower", "stopping")
STATE_FLOATS = ("accel_cmd", *TUNING_FIELDS[:-1])
CONFIG_FIELDS = ("lookahead_jerk_bp", "lookahead_jerk_upper_v", "lookahead_jerk_lower_v", "longitudinal_actuator_delay", "jerk_limits")
STATES = ("off", "pid", "stopping", "starting")
_VERIFIED_RUNTIME = object()


class OutputError(ValueError):
  pass


def _finite(value):
  return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def static_identity(cp, sp):
  return search.value_hash({"cp": cp, "sp": sp})


def source_hashes(repo):
  return {name: hashlib.sha256((Path(repo) / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for name in SOURCES}


def bootstrap_hashes(repo):
  root = Path(repo).resolve()
  result = {}
  for name in BOOTSTRAP:
    path = root / name
    if not path.resolve().is_relative_to(root) or path.is_symlink() or (path.exists() and not path.is_file()):
      raise OutputError("invalid native bootstrap path: " + name)
    result[name] = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() if path.exists() else None
  return result


def native_control(accel, state, active):
  from opendbc.car.structs import car

  return car.CarControl.new_message(longActive=active, actuators={"accel": accel, "longControlState": state})


def native_state(v, a):
  from opendbc.car.structs import car

  return SimpleNamespace(out=car.CarState.new_message(vEgo=v, aEgo=a))


def _check_cached_namespaces(repo):
  roots = {"openpilot": (repo / "openpilot").resolve(), "opendbc": (repo / "opendbc_repo/opendbc").resolve(), "msgq": (repo / "msgq_repo/msgq").resolve()}
  prefixes = {"openpilot": "openpilot", "opendbc": "opendbc_repo/opendbc", "msgq": "msgq_repo/msgq"}
  for name, module in tuple(sys.modules.items()):
    root = roots.get(name.split(".")[0])
    if root is None:
      continue
    file = getattr(module, "__file__", None)
    paths = tuple(getattr(module, "__path__", ()))
    if (
      (not file and not paths)
      or (file and not Path(file).resolve().is_relative_to(root))
      or any(not Path(path).resolve().is_relative_to(root) for path in paths)
    ):
      raise OutputError("cached native namespace origin differs before import: " + name)
    parts = name.split(".")
    initializer = "/".join((prefixes[parts[0]], *parts[1:], "__init__.py"))
    if initializer in BOOTSTRAP and (
      (BOOTSTRAP[initializer] is None and file)
      or (BOOTSTRAP[initializer] is not None and (not file or Path(file).resolve() != (repo / initializer).resolve()))
      or len(paths) != 1
      or Path(paths[0]).resolve() != (repo / initializer).parent.resolve()
    ):
      raise OutputError("cached native bootstrap origin differs before import: " + name)


@dataclass(frozen=True)
class Bindings:
  controller_type: object
  static_builder: object
  control_builder: object
  state_builder: object
  actual_native: bool = False


def load_runtime(repo, expected_sources, expected_gitlink, expected_head, *, expected_bootstrap):
  """Future owner-reviewed POSIX API entry point; no builds, publishers or writes."""
  if os.name != "posix":
    raise OutputError("actual native runtime requires POSIX")
  repo = Path(repo).resolve()
  if expected_gitlink != OPENDBC_PIN or expected_head != SUPERPROJECT_HEAD or expected_sources != PINNED:
    raise OutputError("independent pinned native source/HEAD manifest differs")
  if source_hashes(repo) != expected_sources:
    raise OutputError("native source content differs")
  if expected_bootstrap != BOOTSTRAP or bootstrap_hashes(repo) != expected_bootstrap:
    raise OutputError("native bootstrap manifest/content differs before import")
  env = dict(os.environ)
  for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES"):
    env.pop(key, None)
  for path, expected in ((repo, SUPERPROJECT_HEAD), (repo / "opendbc_repo", OPENDBC_PIN)):
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, env=env)
    if result.returncode != 0 or result.stdout.strip() != expected:
      raise OutputError("actual native source revision differs")
  tree = subprocess.run(["git", "-C", str(repo), "ls-tree", "HEAD", "opendbc_repo"], capture_output=True, text=True, timeout=10, env=env)
  if tree.returncode != 0 or tree.stdout.strip() != f"160000 commit {OPENDBC_PIN}\topendbc_repo":
    raise OutputError("actual superproject opendbc gitlink differs")
  _check_cached_namespaces(repo)
  sys.path[:0] = [str(repo), str(repo / "opendbc_repo")]
  module = importlib.import_module("opendbc.sunnypilot.car.hyundai.longitudinal.controller")
  from opendbc.car import DT_CTRL

  # Decode once through canonical schema imports, without constructing any Params.
  importlib.import_module("openpilot.cereal")
  if DT_CTRL != 0.01:
    raise OutputError("native DT_CTRL contract differs")
  _check_cached_namespaces(repo)
  for name in SOURCES:
    if not name.endswith(".py"):
      continue
    relative = name.removeprefix("opendbc_repo/")
    module_name = relative.removesuffix("/__init__.py") if relative.endswith("/__init__.py") else relative[:-3]
    cached = sys.modules.get(module_name.replace("/", "."))
    if cached is not None and Path(cached.__file__).resolve() != (repo / name).resolve():
      raise OutputError("cached native import origin differs: " + name)
  bindings = Bindings(module.LongitudinalController, search.native_messages, native_control, native_state, True)
  runtime = Runtime(bindings, expected_sources, expected_gitlink, expected_head, repo, bootstrap=expected_bootstrap, _verified=_VERIFIED_RUNTIME)
  runtime._actual_binding_check()
  return runtime


def _state_record(value):
  # Pinned car.capnp ordinals: off=0, pid=1, stopping=2, starting=3.
  # pycapnp enum-module constants are integers; field readers expose raw+name.
  if type(value) is int:
    ordinal, representation = value, "integer"
  elif type(value) is str and value in STATES:
    ordinal, representation = STATES.index(value), "name"
  elif type(value) is getattr(sys.modules.get("capnp.lib.capnp"), "_DynamicEnum", None):
    ordinal, representation = value.raw, "dynamic_reader"
  else:
    raise OutputError("invalid native longitudinal state enum representation")
  if type(ordinal) is not int or not 0 <= ordinal < len(STATES):
    raise OutputError("invalid native longitudinal state enum ordinal")
  name = STATES[ordinal]
  if representation == "dynamic_reader" and str(value) != name:
    raise OutputError("native longitudinal enum name/ordinal disagree")
  return name, ordinal, representation


def _snapshot(controller):
  floats = {key: getattr(controller, key) for key in STATE_FLOATS}
  tuning = {key: getattr(controller.tuning, key) for key in TUNING_FIELDS}
  config = {key: deepcopy(getattr(controller.car_config, key)) for key in CONFIG_FIELDS}
  state, ordinal, representation = _state_record(controller.long_control_state_last)
  if (
    any(not _finite(v) for v in (*floats.values(), *(tuning[k] for k in TUNING_FIELDS[:-1])))
    or type(controller.stopping) is not bool
    or type(tuning["stopping"]) is not bool
    or type(controller.stopping_count) is not int
    or controller.stopping_count < 0
    or state not in STATES
    or type(controller.enabled) is not bool
  ):
    raise OutputError("invalid full native state/tuning report")
  for key, value in config.items():
    values = value if isinstance(value, (list, tuple)) else [value]
    if not values or any(not _finite(v) for v in values):
      raise OutputError("invalid native car configuration: " + key)
  return {
    **deepcopy(floats),
    "stopping": controller.stopping,
    "stopping_count": controller.stopping_count,
    "long_control_state_last": state,
    "long_control_state_last_ordinal": ordinal,
    "long_control_state_last_representation": representation,
    "tuning": deepcopy(tuning),
    "car_config": config,
    "enabled": bool(controller.enabled),
  }


class Runtime:
  def __init__(self, bindings, sources, pin, head, repo=None, *, bootstrap=None, _verified=None):
    self.bindings, self.sources = bindings, MappingProxyType(dict(sources))
    self.bootstrap = MappingProxyType(dict(bootstrap or {}))
    self.pin, self.head, self.repo = pin, head, None if repo is None else Path(repo).resolve()
    self._verified = _verified

  def _actual_binding_check(self):
    if not self.bindings.actual_native:
      return
    b = self.bindings
    if (
      self._verified is not _VERIFIED_RUNTIME
      or self.repo is None
      or self.pin != OPENDBC_PIN
      or self.head != SUPERPROJECT_HEAD
      or dict(self.sources) != PINNED
      or b.static_builder is not search.native_messages
      or b.control_builder is not native_control
      or b.state_builder is not native_state
      or not inspect.isclass(b.controller_type)
      or Path(inspect.getfile(b.controller_type)).resolve() != self.repo / CONTROLLER
      or Path(inspect.getfile(b.controller_type.update)).resolve() != self.repo / CONTROLLER
      or source_hashes(self.repo) != PINNED
      or dict(self.bootstrap) != BOOTSTRAP
      or bootstrap_hashes(self.repo) != BOOTSTRAP
    ):
      raise OutputError("actual native class/factory/source binding differs")

  def create(self, cp_dict, sp_dict, *, static_identity_sha256):
    try:
      search.native.validate_parameters(cp_dict, sp_dict)
      if (
        cp_dict.get("brand") != "hyundai" or type(cp_dict.get("radarUnavailable")) is not bool or type(sp_dict.get("flags")) is not int or sp_dict["flags"] < 0
      ):
        raise OutputError("missing/invalid consumed Hyundai static contract")
      if not re.fullmatch(r"[0-9a-f]{64}", static_identity_sha256) or static_identity(cp_dict, sp_dict) != static_identity_sha256:
        raise OutputError("full CP/SP static identity differs")
      self._actual_binding_check()
      cp, sp = self.bindings.static_builder(deepcopy(cp_dict), deepcopy(sp_dict))
      controller = self.bindings.controller_type(cp, sp)
      if type(controller) is not self.bindings.controller_type or controller.CP is not cp or controller.CP_SP is not sp:
        raise OutputError("native factory returned a different controller/static object")
      return Handle(self, cp, sp, controller, deepcopy(cp_dict), deepcopy(sp_dict), static_identity_sha256)
    except (TypeError, ValueError, KeyError, AttributeError, search.native.ReplayError) as error:
      raise OutputError("static/native construction rejected: " + str(error)) from error


class Handle:
  def __init__(self, runtime, cp, sp, controller, cp_dict, sp_dict, identity):
    self.runtime, self.cp, self.sp, self.controller = runtime, cp, sp, controller
    self.bindings = runtime.bindings
    self.provenance = {"sources": dict(runtime.sources), "bootstrap": dict(runtime.bootstrap), "pin": runtime.pin, "head": runtime.head}
    self.cp_dict, self.sp_dict, self.identity = cp_dict, sp_dict, identity
    self._canonical_static = static_identity(cp.to_dict(), sp.to_dict())
    self.initial = _snapshot(controller)
    if (
      any(self.initial[key] != 0.0 for key in STATE_FLOATS)
      or self.initial["stopping"]
      or self.initial["stopping_count"] != 0
      or self.initial["long_control_state_last"] != "off"
      or any(self.initial["tuning"][key] != (False if key == "stopping" else 0.0) for key in TUNING_FIELDS)
    ):
      raise OutputError("native constructor does not provide the declared cold state")
    self._configuration = search.value_hash(self.initial["car_config"])
    self._expected_state = search.value_hash(self.initial)
    self._update = controller.update
    self.frame, self.last_t, self.held, self.failed = 0, None, 0.0, False
    self._expected_owned_state = self._owned_state()

  def _owned_state(self):
    if (
      type(self.frame) is not int
      or self.frame < 0
      or not _finite(self.held)
      or (self.last_t is not None and not _finite(self.last_t))
      or type(self.failed) is not bool
    ):
      raise OutputError("invalid adapter owned state")
    return search.value_hash({"frame": self.frame, "last_t": self.last_t, "held": self.held, "failed": self.failed})

  def _check_owned(self):
    if self._owned_state() != self._expected_owned_state:
      raise OutputError("adapter owned phase/output/clock/poison changed outside the owned update")

  def _check_static(self, *, check_state=True):
    self._check_owned()
    current = _snapshot(self.controller)
    if (
      self.controller.CP is not self.cp
      or self.controller.CP_SP is not self.sp
      or static_identity(self.cp.to_dict(), self.sp.to_dict()) != self._canonical_static
      or search.value_hash(current["car_config"]) != self._configuration
    ):
      raise OutputError("native static/configuration mutated during trial")
    if check_state and search.value_hash(current) != self._expected_state:
      raise OutputError("native state changed outside the owned update")

  def step(self, *, t, apply_call, requested_accel, long_state, long_active, simulated_v, simulated_a, published_event_valid=None):
    """Model one publication before its optional apply; t is only that loop label."""
    self._check_owned()
    if self.failed:
      raise OutputError("failed native handle cannot continue or reset")
    if not _finite(t) or (self.last_t is not None and t <= self.last_t):
      raise OutputError("loop/publication clock must be finite and strictly increasing")
    if (
      type(apply_call) is not bool
      or type(long_active) is not bool
      or type(long_state) is not str
      or long_state not in STATES
      or any(not _finite(v) for v in (requested_accel, simulated_v, simulated_a))
      or simulated_v < 0
      or (published_event_valid is not None and type(published_event_valid) is not bool)
    ):
      raise OutputError("explicit finite simulated motion/request and typed apply/state/validity required")
    self._check_static()
    before, frame, published = _snapshot(self.controller), self.frame, self.held
    updated = apply_call and frame % 5 == 0
    try:
      if updated:
        cc = self.bindings.control_builder(float(requested_accel), long_state, long_active)
        cs = self.bindings.state_builder(float(simulated_v), float(simulated_a))
        if self.bindings.actual_native and (
          getattr(self._update, "__self__", None) is not self.controller or getattr(self._update, "__func__", None) is not self.bindings.controller_type.update
        ):
          raise OutputError("actual native update method was replaced")
        self._update(cc, cs)
      after = _snapshot(self.controller)
      self._check_static(check_state=False)
    except Exception as error:
      self.failed = True
      self._expected_owned_state = self._owned_state()
      raise OutputError("native update/state failed; no reset attempted: " + str(error)) from error
    if apply_call:
      self.frame += 1
      self.held = float(after["tuning"]["actual_accel"])
    self.last_t = float(t)
    self._expected_state = search.value_hash(after)
    self._expected_owned_state = self._owned_state()
    return {
      "t": float(t),
      "apply_call": apply_call,
      "apply_frame_before": frame,
      "apply_frame_after": self.frame,
      "native_updated": updated,
      "published_accel": published,
      "held_accel_after_apply": self.held,
      "published_event_valid": published_event_valid,
      "requested_accel": float(requested_accel),
      "long_state": long_state,
      "long_active": long_active,
      "simulated_v": float(simulated_v),
      "simulated_a": float(simulated_a),
      "native_before": before,
      "native_after": after,
      "static_identity_sha256": self.identity,
      "actual_native": self.bindings.actual_native,
      "initialization_uncertain": True,
      "safety_acceptance": "not_evaluated",
      "delivered_force": "not_identified",
      "selection_eligible": False,
      "production_enabled": False,
    }

  def receipt(self):
    self._check_static()
    return {
      "static_cp": deepcopy(self.cp_dict),
      "static_sp": deepcopy(self.sp_dict),
      "static_identity_sha256": self.identity,
      "source_hashes": dict(self.provenance["sources"]),
      "bootstrap_sources": dict(self.provenance["bootstrap"]),
      "opendbc_gitlink": self.provenance["pin"],
      "superproject_head": self.provenance["head"],
      "actual_native": self.bindings.actual_native,
      "initialization_uncertain": True,
      "cold_initial_state": deepcopy(self.initial),
      "native_state": _snapshot(self.controller),
      "apply_calls": self.frame,
      "held_output": self.held,
      "failed": self.failed,
      "publication_order": "previous held report before optional current apply; no fixed delay assumption",
      "publication_validity": "caller-supplied or unknown; not derived from apply_call",
      "safety_acceptance": "not_evaluated",
      "delivered_force": "not_identified",
      "selection_eligible": False,
      "research_helpers": {
        name: hashlib.sha256(Path(inspect.getfile(fn)).read_bytes()).hexdigest()
        for name, fn in (("native_messages", search.native_messages), ("validate_parameters", search.native.validate_parameters))
      },
    }
