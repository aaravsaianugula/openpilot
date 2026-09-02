"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Resolves which published driving-model catalog this build can actually run.

The catalogs are versioned files on gh-pages, and a new version is published whenever the
models are recompiled. Hardcoding one filename strands the device on a stale list: the eGPU
feed was renamed `usbgpu` -> `chestnut` and moved on two versions, and the device kept
fetching the old name. So we look for the newest published version instead of naming one.

A newer catalog is only usable if its models were compiled against the tinygrad this build
runs, and if its bundles declare a selector version this build understands. Both are carried
in the catalog itself, so the newest *runnable* one can be picked without a code change --
and a catalog we could not execute is refused rather than silently downloaded.

Deliberately free of openpilot imports: this must be testable off-device.
"""

import time
from collections.abc import Callable
from typing import NamedTuple

BASE_URL = "https://raw.githubusercontent.com/sunnypilot/sunnypilot-models/refs/heads/gh-pages/docs/"

# Current name first. The eGPU feed was renamed usbgpu -> chestnut; keeping the old name in
# the search means the next rename does not strand the device either.
ONBOARD_FAMILIES = ("driving_models_v{n}.json",)
CHESTNUT_FAMILIES = ("driving_models_chestnut_v{n}.json", "driving_models_usbgpu_v{n}.json")

# Newest catalog this build shipped against. Never look below it: older catalogs are not
# an improvement, and probing down would only cost requests.
ONBOARD_FLOOR = 21
CHESTNUT_FLOOR = 22

# Versions probed per family, starting at the floor.
MAX_LOOKAHEAD = 6
# Stop a family after this many misses in a row, so a single unpublished version does not
# hide a newer one but a permanent gap ends the walk.
MAX_CONSECUTIVE_MISSES = 2


class Catalog(NamedTuple):
  url: str
  version: int
  data: dict


def families(is_chestnut: bool) -> tuple[str, ...]:
  return CHESTNUT_FAMILIES if is_chestnut else ONBOARD_FAMILIES


def floor(is_chestnut: bool) -> int:
  return CHESTNUT_FLOOR if is_chestnut else ONBOARD_FLOOR


def floor_url(is_chestnut: bool) -> str:
  """The catalog this build shipped against -- the fallback when nothing resolves."""
  return BASE_URL + families(is_chestnut)[0].format(n=floor(is_chestnut))


def _selector_version(bundle: dict) -> int:
  """The published feed carries this as a string; -1 means 'not a version we can read'."""
  try:
    return int(bundle.get("minimum_selector_version"))
  except (AttributeError, TypeError, ValueError):
    return -1


def catalog_is_usable(data: dict | None, tinygrad_ref: str | None, required_version: int) -> bool:
  if not isinstance(data, dict):
    return False

  bundles = data.get("bundles")
  if not isinstance(bundles, list) or not bundles:
    return False

  # Artifacts are compiled per tinygrad commit. A catalog built against another one lists
  # models this build cannot execute, so it is not a candidate however new it is.
  # An unknown ref refuses everything rather than skipping the check: get_tinygrad_ref()
  # returns None on any error, and treating that as "no opinion" would admit exactly the
  # mismatched catalog this exists to keep out. Nothing resolves, so the floor URL is used.
  if not tinygrad_ref or data.get("tinygrad_ref") != tinygrad_ref:
    return False

  return any(isinstance(b, dict) and _selector_version(b) == required_version for b in bundles)


def _published_versions(is_chestnut: bool, head_ok: Callable[[str], bool]) -> list[tuple[int, str]]:
  """(version, url) for every catalog that exists, newest first, current name first on a tie."""
  found: list[tuple[int, str]] = []
  start = floor(is_chestnut)

  for family in families(is_chestnut):
    misses = 0
    for n in range(start, start + MAX_LOOKAHEAD):
      url = BASE_URL + family.format(n=n)
      if head_ok(url):
        found.append((n, url))
        misses = 0
      else:
        misses += 1
        if misses >= MAX_CONSECUTIVE_MISSES:
          break

  # stable, so families keep their declared order within a version
  found.sort(key=lambda vu: -vu[0])
  return found


def resolve_catalog(*, is_chestnut: bool, tinygrad_ref: str | None, required_version: int,
                    head_ok: Callable[[str], bool],
                    fetch_json: Callable[[str], dict | None]) -> Catalog | None:
  """Newest published catalog this build can run, or None if none of them qualify.

  Existence is probed with cheap HEADs; only catalogs that exist are downloaded, newest
  first, and the first usable one wins.
  """
  for version, url in _published_versions(is_chestnut, head_ok):
    data = fetch_json(url)
    if catalog_is_usable(data, tinygrad_ref, required_version):
      return Catalog(url, version, data)
  return None


# A failed resolve must not be retried on every manager tick: the daemon runs at 1 Hz and a
# device with no connectivity would otherwise issue a HEAD storm over the car's LTE link.
RESOLVE_RETRY_S = 300.0


class CatalogResolver:
  """Resolves once per context and remembers it, with a backoff on failure.

  Success is cached for the life of the process -- the catalog does not change under a
  running device, and re-probing would cost requests for nothing. Failure is cached only
  until the retry interval elapses, so a device that boots with no connection still picks
  up the newest catalog once it gets one.
  """

  def __init__(self, head_ok: Callable[[str], bool], fetch_json: Callable[[str], dict | None],
               tinygrad_ref: str | None, required_version: int,
               monotonic: Callable[[], float] | None = None):
    self._head_ok = head_ok
    self._fetch_json = fetch_json
    self._tinygrad_ref = tinygrad_ref
    self._required_version = required_version
    self._monotonic = monotonic or time.monotonic
    self._resolved: dict[bool, Catalog] = {}
    self._failed_at: dict[bool, float] = {}

  def resolve(self, is_chestnut: bool) -> Catalog | None:
    """The chosen catalog, or None while we have not found a usable one."""
    if (hit := self._resolved.get(is_chestnut)) is not None:
      return hit

    now = self._monotonic()
    last_failure = self._failed_at.get(is_chestnut)
    if last_failure is not None and (now - last_failure) < RESOLVE_RETRY_S:
      return None

    found = resolve_catalog(is_chestnut=is_chestnut, tinygrad_ref=self._tinygrad_ref,
                            required_version=self._required_version,
                            head_ok=self._head_ok, fetch_json=self._fetch_json)
    if found is None:
      self._failed_at[is_chestnut] = now
      return None

    self._failed_at.pop(is_chestnut, None)
    self._resolved[is_chestnut] = found
    return found
