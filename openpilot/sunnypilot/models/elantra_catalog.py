"""
Bundles we host ourselves, merged into sunnypilot's catalog for the same source.

Why this exists. sunnypilot's catalog is the only way a model reaches the Models screen. There
is no local source, no sideload path and no custom-bundle param: `ModelFetcher` has two hardcoded
URLs, the picker lists what they contain, and `validate_active_bundles` clears any active bundle
that is not in the fetched list. So carrying a driving model ahead of sunnypilot -- as this port
does for comma's cinque terre model -- means adding an entry to the list they serve.

The merge happens on the fetched document before it is cached, which is the one point every
reader passes through: the manager resolves and downloads from `get_bundles_for_source`, the UI
lists from `get_cached_bundles`, and both land on that cache.

This module deliberately imports nothing but the standard library at module scope (`requests` is
imported inside the fetch function). That keeps `merge_extra_bundles` testable on a bare Python 3
with no cereal and no capnp, which is what `.elantra` tests and the sync runner have.

Two rules this file exists to hold:

  * **fail open.** An unreachable, unpublished or malformed catalog of ours must leave
    sunnypilot's list exactly as it arrived. Our models must never be able to empty the picker
    or invalidate someone's selection.
  * **defer to sunnypilot.** Once they publish the same `ref`, theirs wins and ours drops out,
    so the driver never sees two entries for one model pointing at different hosts.
"""

from __future__ import annotations

# Where our own catalog document lives. A GitHub release asset on the fork: stable per tag,
# plain HTTPS, and the same host already serves the chunks it points at. Absent until a model
# is actually published, and a 404 here is a normal state, not an error.
EXTRA_MODEL_URLS = {
  "chestnut": "https://github.com/aaravsaianugula/openpilot/releases/download/elantra-models/elantra_models_chestnut.json",
}

EXTRA_FETCH_TIMEOUT = 10


def _bundle_list(document) -> list:
  """The bundles array, or [] for anything that is not a well-formed document."""
  if not isinstance(document, dict):
    return []
  bundles = document.get("bundles")
  return bundles if isinstance(bundles, list) else []


def merge_extra_bundles(json_data, extra_json):
  """Return `json_data` with our bundles appended. Never raises, never mutates the input.

  Both arguments are whatever came off the wire, so both are treated as untrusted shapes.
  """
  additions = [b for b in _bundle_list(extra_json)
               if isinstance(b, dict) and b.get("ref")]
  if not additions:
    return json_data

  existing = _bundle_list(json_data)
  if not isinstance(json_data, dict):
    # Nothing sane to merge into. Hand back exactly what we were given rather than
    # inventing a document the caller would then cache.
    return json_data

  known_refs = {b.get("ref") for b in existing if isinstance(b, dict)}

  # sunnypilot's own entry wins on a ref collision: when they publish a model we were
  # carrying early, ours must disappear rather than shadow theirs.
  additions = [b for b in additions if b["ref"] not in known_refs]
  if not additions:
    return json_data

  # Their chestnut list is 0..10 today and grows. Reusing a live index reorders or hides one
  # of their models, so ours are renumbered above whatever the fetched document uses.
  indices = [b.get("index") for b in existing if isinstance(b, dict)]
  highest = max((i for i in indices if isinstance(i, int)), default=-1)

  merged = dict(json_data)
  renumbered = []
  for offset, b in enumerate(additions):
    entry = dict(b)
    entry["index"] = highest + 1 + offset
    renumbered.append(entry)

  merged["bundles"] = list(existing) + renumbered
  return merged


def fetch_extra_json(url: str, timeout: int = EXTRA_FETCH_TIMEOUT):
  """Fetch our catalog document. Returns None for every failure, including 'not published yet'."""
  import requests

  try:
    response = requests.get(url, timeout=timeout)
    if response.status_code == 404:
      return None
    response.raise_for_status()
    return response.json()
  except Exception:
    return None


def merge_for_source(source: str, json_data):
  """Merge our bundles for `source`, if we host any. Safe to call on every fetch."""
  url = EXTRA_MODEL_URLS.get(source)
  if not url:
    return json_data
  return merge_extra_bundles(json_data, fetch_extra_json(url))
