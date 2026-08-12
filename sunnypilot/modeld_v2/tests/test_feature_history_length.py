import pytest

from openpilot.sunnypilot.modeld_v2.compile_modeld import _feat_q_len


# (1, 32, 512) and frame_skip 4 are RDF checkpoint 5's declared values.
RDF_FEATURES_BUFFER = (1, 32, 512)
RDF_FRAME_SKIP = 4


def test_openpilot_experiment_layout_spans_the_full_history():
  # The policy JIT in a format_version 1 artifact was captured against
  # (128, 1, 512); anything else fails at call time with an args mismatch,
  # not with a wrong number.
  assert _feat_q_len(RDF_FEATURES_BUFFER, RDF_FRAME_SKIP, True) == 128


def test_sunnypilot_layout_spans_the_gaps_between_entries():
  assert _feat_q_len(RDF_FEATURES_BUFFER, RDF_FRAME_SKIP, False) == 125


def test_the_two_conventions_differ_whenever_frames_are_skipped():
  # They coincide only at frame_skip 1, which is why a single formula looked
  # correct until a skipping model showed up.
  assert _feat_q_len(RDF_FEATURES_BUFFER, 1, True) == _feat_q_len(RDF_FEATURES_BUFFER, 1, False)
  for skip in (2, 3, 4, 5):
    assert _feat_q_len(RDF_FEATURES_BUFFER, skip, True) != _feat_q_len(RDF_FEATURES_BUFFER, skip, False)


@pytest.mark.parametrize("features_buffer,skip,expected", [
  ((1, 24, 512), 4, 96),
  ((1, 25, 512), 2, 50),
  ((1, 32, 512), 4, 128),
])
def test_full_history_is_exactly_entries_times_skip(features_buffer, skip, expected):
  assert _feat_q_len(features_buffer, skip, True) == expected
