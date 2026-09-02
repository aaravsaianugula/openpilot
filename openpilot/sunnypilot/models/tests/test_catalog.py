"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Catalog resolution has no openpilot dependencies on purpose: it must be runnable on a
workstation, not only on the device.
"""

import os
import pathlib
import unittest

from openpilot.sunnypilot.models import catalog

REF_OURS = "66ee3cfb4f3a3908a6a20ddfbec7774ba7c09b4e"
REF_NEWER = "e837e367aac9000000000000000000000000000d"
VERSION_OURS = 18


def make_catalog(tinygrad_ref: str, selector_version: int, n_bundles: int = 2) -> dict:
  return {
    "tinygrad_ref": tinygrad_ref,
    "bundles": [{"index": i, "short_name": f"M{i}", "minimum_selector_version": str(selector_version)}
                for i in range(n_bundles)],
  }


class FakeHost:
  """Serves a fixed {url: json} map and counts every request."""

  def __init__(self, pages: dict[str, dict]):
    self.pages = pages
    self.heads: list[str] = []
    self.gets: list[str] = []

  def head_ok(self, url: str) -> bool:
    self.heads.append(url)
    return url in self.pages

  def fetch_json(self, url: str) -> dict | None:
    self.gets.append(url)
    return self.pages.get(url)


def url_for(name: str) -> str:
  return catalog.BASE_URL + name


def resolve(host: FakeHost, is_chestnut: bool = True, tinygrad_ref: str = REF_OURS,
            required_version: int = VERSION_OURS):
  return catalog.resolve_catalog(is_chestnut=is_chestnut, tinygrad_ref=tinygrad_ref,
                                 required_version=required_version,
                                 head_ok=host.head_ok, fetch_json=host.fetch_json)


class TestCatalogResolve(unittest.TestCase):
  def test_picks_highest_compatible_version(self):
    host = FakeHost({
      url_for("driving_models_chestnut_v22.json"): make_catalog(REF_OURS, VERSION_OURS),
      url_for("driving_models_chestnut_v23.json"): make_catalog(REF_OURS, VERSION_OURS),
    })
    found = resolve(host)
    assert found is not None
    assert found.version == 23
    assert found.url == url_for("driving_models_chestnut_v23.json")

  def test_rejects_catalog_built_against_another_tinygrad(self):
    """The v23 case in the wild: newer catalog, artifacts recompiled against a tinygrad we
    do not run. Following it would download models this build cannot execute."""
    host = FakeHost({
      url_for("driving_models_chestnut_v22.json"): make_catalog(REF_OURS, VERSION_OURS),
      url_for("driving_models_chestnut_v23.json"): make_catalog(REF_NEWER, VERSION_OURS),
    })
    found = resolve(host)
    assert found is not None
    assert found.version == 22, "must fall back to the newest catalog matching our tinygrad"

  def test_rejects_incompatible_selector_version(self):
    host = FakeHost({
      url_for("driving_models_chestnut_v22.json"): make_catalog(REF_OURS, VERSION_OURS),
      url_for("driving_models_chestnut_v23.json"): make_catalog(REF_OURS, VERSION_OURS + 1),
    })
    found = resolve(host)
    assert found is not None
    assert found.version == 22

  def test_returns_none_when_nothing_qualifies(self):
    host = FakeHost({
      url_for("driving_models_chestnut_v22.json"): make_catalog(REF_NEWER, VERSION_OURS),
    })
    assert resolve(host) is None

  def test_returns_none_when_host_is_unreachable(self):
    assert resolve(FakeHost({})) is None

  def test_follows_the_usbgpu_to_chestnut_rename(self):
    """Only the legacy name is published: the device must still find its models."""
    host = FakeHost({
      url_for("driving_models_usbgpu_v22.json"): make_catalog(REF_OURS, VERSION_OURS),
    })
    found = resolve(host)
    assert found is not None
    assert found.url == url_for("driving_models_usbgpu_v22.json")

  def test_prefers_the_current_name_on_a_version_tie(self):
    host = FakeHost({
      url_for("driving_models_usbgpu_v22.json"): make_catalog(REF_OURS, VERSION_OURS),
      url_for("driving_models_chestnut_v22.json"): make_catalog(REF_OURS, VERSION_OURS),
    })
    found = resolve(host)
    assert found is not None
    assert found.url == url_for("driving_models_chestnut_v22.json")

  def test_onboard_family_is_separate_from_chestnut(self):
    host = FakeHost({
      url_for("driving_models_v21.json"): make_catalog(REF_OURS, VERSION_OURS),
      url_for("driving_models_chestnut_v22.json"): make_catalog(REF_OURS, VERSION_OURS),
    })
    found = resolve(host, is_chestnut=False)
    assert found is not None
    assert found.url == url_for("driving_models_v21.json")

  def test_never_probes_below_the_floor(self):
    host = FakeHost({url_for("driving_models_chestnut_v22.json"): make_catalog(REF_OURS, VERSION_OURS)})
    resolve(host)
    versions = [int(u.rsplit("_v", 1)[1].removesuffix(".json")) for u in host.heads]
    assert min(versions) == catalog.CHESTNUT_FLOOR

  def test_probe_is_bounded_when_the_feed_runs_far_ahead(self):
    pages = {url_for(f"driving_models_chestnut_v{n}.json"): make_catalog(REF_OURS, VERSION_OURS)
             for n in range(catalog.CHESTNUT_FLOOR, catalog.CHESTNUT_FLOOR + 50)}
    host = FakeHost(pages)
    found = resolve(host)
    assert found is not None
    assert len(host.heads) <= 2 * catalog.MAX_LOOKAHEAD, f"unbounded probe: {len(host.heads)} requests"
    assert found.version == catalog.CHESTNUT_FLOOR + catalog.MAX_LOOKAHEAD - 1

  def test_a_gap_does_not_stop_the_walk(self):
    """One missing version between two published ones must not hide the newer one."""
    host = FakeHost({
      url_for("driving_models_chestnut_v22.json"): make_catalog(REF_OURS, VERSION_OURS),
      url_for("driving_models_chestnut_v24.json"): make_catalog(REF_OURS, VERSION_OURS),
    })
    found = resolve(host)
    assert found is not None
    assert found.version == 24

  def test_body_is_only_fetched_for_versions_that_exist(self):
    host = FakeHost({url_for("driving_models_chestnut_v22.json"): make_catalog(REF_OURS, VERSION_OURS)})
    resolve(host)
    assert host.gets == [url_for("driving_models_chestnut_v22.json")], host.gets

  def test_empty_catalog_is_rejected(self):
    host = FakeHost({url_for("driving_models_chestnut_v22.json"): {"tinygrad_ref": REF_OURS, "bundles": []}})
    assert resolve(host) is None

  def test_malformed_catalog_is_rejected(self):
    host = FakeHost({url_for("driving_models_chestnut_v22.json"): {"bundles": "not-a-list"}})
    assert resolve(host) is None

  def test_unparseable_selector_version_is_rejected(self):
    host = FakeHost({url_for("driving_models_chestnut_v22.json"): {
      "tinygrad_ref": REF_OURS,
      "bundles": [{"minimum_selector_version": "eighteen"}],
    }})
    assert resolve(host) is None

  def test_an_unreadable_tinygrad_ref_refuses_everything(self):
    """get_tinygrad_ref() swallows every error and returns None. If an unknown ref merely
    skipped the check, a broken submodule read would silently admit a catalog compiled
    against another tinygrad -- the exact download this check exists to prevent."""
    host = FakeHost({
      url_for("driving_models_chestnut_v22.json"): make_catalog(REF_OURS, VERSION_OURS),
      url_for("driving_models_chestnut_v23.json"): make_catalog(REF_NEWER, VERSION_OURS),
    })
    assert resolve(host, tinygrad_ref=None) is None
    assert resolve(host, tinygrad_ref="") is None

  def test_floors_match_the_urls_this_build_shipped_with(self):
    assert catalog.floor_url(is_chestnut=False).endswith(f"driving_models_v{catalog.ONBOARD_FLOOR}.json")
    assert catalog.floor_url(is_chestnut=True).endswith(f"driving_models_chestnut_v{catalog.CHESTNUT_FLOOR}.json")


class TestBrowsingDoesNotChangeWhatRuns(unittest.TestCase):
  """The toggle exists to browse and pre-download. It must not re-activate a parked eGPU
  bundle: validate_active_bundle's two-slot stash is keyed on the hardware context, so
  running it while merely browsing swaps the model that will actually drive."""

  @staticmethod
  def validates(chestnut_present: bool, show_toggle: bool) -> bool:
    # mirrors manager.main_thread
    use_chestnut = chestnut_present or show_toggle
    browsing_only = use_chestnut and not chestnut_present
    return not browsing_only

  def test_hardware_contexts_still_validate(self):
    assert self.validates(chestnut_present=False, show_toggle=False), "on-board must still validate"
    assert self.validates(chestnut_present=True, show_toggle=False), "dock attached must still validate"
    assert self.validates(chestnut_present=True, show_toggle=True), "dock wins over the toggle"

  def test_browsing_without_the_dock_does_not_validate(self):
    assert not self.validates(chestnut_present=False, show_toggle=True)

  def test_manager_implements_exactly_this(self):
    src = (pathlib.Path(__file__).parent.parent / "manager.py").read_text(encoding="utf-8")
    assert "browsing_only" in src, "manager no longer guards validation while browsing"
    assert "not browsing_only" in src


class FakeClock:
  def __init__(self):
    self.t = 1000.0

  def __call__(self) -> float:
    return self.t

  def advance(self, seconds: float) -> None:
    self.t += seconds


class TestCatalogResolver(unittest.TestCase):
  """The manager calls this at 1 Hz. Anything it does per call, it does 3600 times an hour."""

  def make(self, host, clock=None):
    return catalog.CatalogResolver(head_ok=host.head_ok, fetch_json=host.fetch_json,
                                   tinygrad_ref=REF_OURS, required_version=VERSION_OURS,
                                   monotonic=clock or FakeClock())

  def test_success_is_resolved_once_not_once_per_call(self):
    host = FakeHost({url_for("driving_models_chestnut_v22.json"): make_catalog(REF_OURS, VERSION_OURS)})
    resolver = self.make(host)
    first = resolver.resolve(True)
    requests_after_first = len(host.heads) + len(host.gets)
    for _ in range(50):
      assert resolver.resolve(True) is first
    assert len(host.heads) + len(host.gets) == requests_after_first, "re-probed a catalog it already had"

  def test_failure_backs_off_instead_of_retrying_every_call(self):
    """A device with no connectivity must not issue a HEAD storm over the car's link."""
    host = FakeHost({})
    clock = FakeClock()
    resolver = self.make(host, clock)
    assert resolver.resolve(True) is None
    attempted = len(host.heads)
    assert attempted > 0

    for _ in range(100):
      assert resolver.resolve(True) is None
    assert len(host.heads) == attempted, f"retried while backing off: {len(host.heads)} vs {attempted}"

  def test_retry_happens_once_the_backoff_elapses(self):
    host = FakeHost({})
    clock = FakeClock()
    resolver = self.make(host, clock)
    assert resolver.resolve(True) is None
    attempted = len(host.heads)

    clock.advance(catalog.RESOLVE_RETRY_S + 1)
    host.pages[url_for("driving_models_chestnut_v22.json")] = make_catalog(REF_OURS, VERSION_OURS)
    found = resolver.resolve(True)
    assert found is not None, "a device that boots offline must still resolve once it is online"
    assert len(host.heads) > attempted

  def test_contexts_are_resolved_independently(self):
    host = FakeHost({
      url_for("driving_models_v21.json"): make_catalog(REF_OURS, VERSION_OURS),
      url_for("driving_models_chestnut_v22.json"): make_catalog(REF_OURS, VERSION_OURS),
    })
    resolver = self.make(host)
    onboard = resolver.resolve(False)
    chestnut = resolver.resolve(True)
    assert onboard is not None and chestnut is not None
    assert onboard.url != chestnut.url


@unittest.skipUnless(os.environ.get('RUN_INTEGRATION_TESTS'), 'requires external network')
class TestCatalogResolveLive(unittest.TestCase):
  """What the device will actually resolve to, against the published feeds."""

  def _resolve(self, is_chestnut: bool):
    import requests

    def head_ok(url):
      return requests.head(url, timeout=15).status_code == 200

    def fetch_json(url):
      r = requests.get(url, timeout=30)
      return r.json() if r.status_code == 200 else None

    from openpilot.sunnypilot.models.tinygrad_ref import get_tinygrad_ref
    return catalog.resolve_catalog(is_chestnut=is_chestnut, tinygrad_ref=get_tinygrad_ref(),
                                   required_version=VERSION_OURS, head_ok=head_ok, fetch_json=fetch_json)

  def test_chestnut_resolves_to_a_runnable_catalog(self):
    found = self._resolve(True)
    assert found is not None, "no chestnut catalog matches our tinygrad ref"
    print(f"chestnut -> {found.url} ({len(found.data['bundles'])} bundles)")

  def test_onboard_resolves_to_a_runnable_catalog(self):
    found = self._resolve(False)
    assert found is not None, "no on-board catalog matches our tinygrad ref"
    print(f"on-board -> {found.url} ({len(found.data['bundles'])} bundles)")


if __name__ == '__main__':
  unittest.main()
