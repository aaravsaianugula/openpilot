import pytest

from openpilot.sunnypilot.models.contracts import (APPROVED_ARTIFACT_HOSTS, ArtifactSourceError,
                                                  validate_artifact_url)
from openpilot.sunnypilot.models.openpilot_experimental import OPENPILOT_EXPERIMENTAL_BUNDLES


@pytest.mark.parametrize("url", [
  "https://gitlab.com/sunnypilot/public/docs.sunnypilot.ai/-/raw/main/driving.pkl",
  "https://raw.githubusercontent.com/firestar5683/StarPilot-Resources/Models/rdf53_driving_tinygrad.pkl",
  "https://RAW.GithubUserContent.com/x/y.pkl",
])
def test_accepts_approved_https_hosts(url):
  assert validate_artifact_url(url) == url


@pytest.mark.parametrize("url", [
  "http://raw.githubusercontent.com/x/y.pkl",             # plaintext
  "https://evil.example.com/y.pkl",                       # unapproved host
  "https://raw.githubusercontent.com.evil.example/y.pkl",  # suffix-confusion
  "https://user:pw@raw.githubusercontent.com/y.pkl",      # embedded credentials
  "file:///etc/passwd",
  "",
  None,
])
def test_rejects_everything_else(url):
  with pytest.raises(ArtifactSourceError):
    validate_artifact_url(url)


def test_shipped_artifact_urls_are_on_the_approved_path():
  for bundle in OPENPILOT_EXPERIMENTAL_BUNDLES:
    artifact = bundle["models"][0]["artifact"]
    assert validate_artifact_url(artifact["download_uri"]["url"])
    assert artifact["download_uri"]["sha256"], bundle["short_name"]


def test_approved_hosts_are_not_silently_widened():
  # Widening this set widens what may deliver code into the driving loop.
  assert APPROVED_ARTIFACT_HOSTS == {"gitlab.com", "raw.githubusercontent.com"}
