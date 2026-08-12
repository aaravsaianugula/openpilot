"""Trust boundary for driving-model artifact downloads.

A driving model steers the car, and it is delivered as a pickle that the model
runtime unpickles - so whatever the catalog names as a download URL decides what
code runs. Two controls apply, and the second is the load-bearing one:

  1. the artifact must be served over HTTPS by a host on this list, and
  2. the reassembled file must match a SHA-256 the catalog pins.

For the OpenPilot experimental bundles that digest is pinned in this repository
(`openpilot_experimental.py`), so the bytes are fixed to what was reviewed rather
than to whatever the third-party publisher serves today. The host list on its own
is weak - anyone can publish under an approved host - which is why an artifact
without a pinned digest is not made safe by passing this check.
"""

from urllib.parse import urlparse


APPROVED_ARTIFACT_HOSTS = frozenset({"gitlab.com", "raw.githubusercontent.com"})


class ArtifactSourceError(ValueError):
  """Raised when an artifact URL is not on the approved download path."""


def validate_artifact_url(url: str) -> str:
  """Return `url` if it is a fetchable artifact source, else raise.

  Applied per URL rather than per bundle because chunk names are resolved with
  `urljoin`, and an absolute chunk name would otherwise redirect the download to
  an arbitrary host.
  """
  parsed = urlparse(url or "")
  if parsed.scheme != "https":
    raise ArtifactSourceError(f"artifact url must use https: {url!r}")
  if not parsed.hostname or parsed.hostname.lower() not in APPROVED_ARTIFACT_HOSTS:
    raise ArtifactSourceError(f"artifact host is not approved: {url!r}")
  if parsed.username or parsed.password:
    raise ArtifactSourceError("artifact url must not carry credentials")
  return url
