import pytest

from openpilot.sunnypilot.models import manager as model_manager
from openpilot.sunnypilot.models.manager import ModelManagerSP
from openpilot.sunnypilot.models.openpilot_experimental import DEFAULT_BUNDLE_INDEX


class FakeParams:
  """Params double that keeps the real API's type discipline.

  The Windows test conftest stubs Params with a permissive fake whose `get`
  returns None for everything, which would let a wrong call signature or a
  non-bool write pass here and only fail on the device. This one rejects both.
  """

  def __init__(self, initial: dict | None = None):
    self._values = dict(initial or {})

  def get(self, key, block=False, return_default=False):
    return self._values.get(key)

  def get_bool(self, key):
    value = self._values.get(key, False)
    if not isinstance(value, bool):
      raise TypeError(f"{key} holds {type(value).__name__}, not bool")
    return value

  def put(self, key, value, block=False):
    self._values[key] = value

  def put_bool(self, key, value, block=False):
    if not isinstance(value, bool):
      raise TypeError(f"{key} must be written as bool, got {type(value).__name__}")
    self._values[key] = value

  def remove(self, key):
    self._values.pop(key, None)


class FakeBundle:
  def __init__(self, index):
    self.index = index


def make_manager(params, available=(), active=None):
  # Bypass __init__: it builds a real PubMaster, which this path does not touch.
  manager = ModelManagerSP.__new__(ModelManagerSP)
  manager.params = params
  manager.available_models = list(available)
  manager.active_bundle = active
  return manager


@pytest.fixture(autouse=True)
def _quiet_cloudlog(monkeypatch):
  monkeypatch.setattr(model_manager.cloudlog, "info", lambda *_a, **_kw: None)


def catalog_with_default():
  return [FakeBundle(73), FakeBundle(DEFAULT_BUNDLE_INDEX), FakeBundle(1006)]


def test_seeds_the_default_on_a_device_that_has_never_chosen():
  params = FakeParams()
  manager = make_manager(params, catalog_with_default())

  manager._seed_default_bundle()

  assert params.get("ModelManager_DownloadIndex") == DEFAULT_BUNDLE_INDEX


def test_does_not_seed_once_the_driver_has_chosen():
  # Covers picking "Default" (stock): ActiveBundle is cleared, but the choice
  # must still stand instead of being re-seeded on the next poll.
  params = FakeParams({"ModelManager_UserChoseModel": True})
  manager = make_manager(params, catalog_with_default())

  manager._seed_default_bundle()

  assert params.get("ModelManager_DownloadIndex") is None


def test_does_not_seed_when_a_bundle_is_already_active():
  params = FakeParams()
  manager = make_manager(params, catalog_with_default(), active=FakeBundle(73))

  manager._seed_default_bundle()

  assert params.get("ModelManager_DownloadIndex") is None


def test_does_not_seed_before_the_catalog_is_available():
  # A device that has never been online has no catalog yet; seeding must wait
  # rather than burn the one chance to select the default.
  params = FakeParams()
  manager = make_manager(params, [])

  manager._seed_default_bundle()

  assert params.get("ModelManager_DownloadIndex") is None


def test_retries_on_a_later_poll_after_the_catalog_arrives():
  params = FakeParams()
  offline = make_manager(params, [])
  offline._seed_default_bundle()
  assert params.get("ModelManager_DownloadIndex") is None

  online = make_manager(params, catalog_with_default())
  online._seed_default_bundle()

  assert params.get("ModelManager_DownloadIndex") == DEFAULT_BUNDLE_INDEX


def test_does_not_disturb_a_download_already_in_flight():
  params = FakeParams({"ModelManager_DownloadIndex": 73})
  manager = make_manager(params, catalog_with_default())

  manager._seed_default_bundle()

  assert params.get("ModelManager_DownloadIndex") == 73


def test_seeding_never_writes_an_active_bundle_itself():
  # Activation stays with the download path, which only promotes a bundle after
  # its artifact matches the pinned digest. Seeding must not shortcut that.
  params = FakeParams()
  manager = make_manager(params, catalog_with_default())

  manager._seed_default_bundle()

  assert params.get("ModelManager_ActiveBundle") is None
