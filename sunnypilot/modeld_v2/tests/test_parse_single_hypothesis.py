"""The OpenPilot experimental models emit single-hypothesis plan and lead.

This is the case that reached the car: RDF v5 loaded, ran, and produced finite
outputs, then modeld died in the parser because `plan` was read as a 5-way
mixture. Driving the raw JIT is not enough - the parser has to be exercised on
the real output widths.
"""

import numpy as np
import pytest

from openpilot.sunnypilot.modeld_v2.constants import ModelConstants
from openpilot.sunnypilot.modeld_v2.parse_model_outputs import Parser


# Widths taken from rdf53_driving_tinygrad.pkl's declared output_slices.
RDF_WIDTHS = {
  'lane_lines': 528, 'lane_lines_prob': 8, 'road_edges': 264, 'meta': 55,
  'desire_pred': 32, 'pose': 12, 'wide_from_device_euler': 6, 'road_transform': 12,
  'plan': 990, 'lead': 144, 'lead_prob': 3, 'desire_state': 8,
}

# Same set as a sunnypilot-style mixture model, for the other branch.
MHP_WIDTHS = dict(RDF_WIDTHS, plan=4955, lead=102)


def make_outs(widths):
  return {k: np.zeros((1, w), dtype=np.float32) for k, w in widths.items()}


def test_single_hypothesis_plan_and_lead_parse():
  outs = Parser().parse_outputs(make_outs(RDF_WIDTHS))

  assert outs['plan'].shape == (1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH)
  assert outs['lead'].shape == (1, ModelConstants.LEAD_MHP_SELECTION,
                                ModelConstants.LEAD_TRAJ_LEN, ModelConstants.LEAD_WIDTH)


def test_mixture_plan_and_lead_still_parse():
  outs = Parser().parse_outputs(make_outs(MHP_WIDTHS))

  # Same downstream shapes either way, so nothing after the parser has to care.
  assert outs['plan'].shape == (1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH)
  assert outs['lead'].shape == (1, ModelConstants.LEAD_MHP_SELECTION,
                                ModelConstants.LEAD_TRAJ_LEN, ModelConstants.LEAD_WIDTH)
  # Only the mixture form exposes per-hypothesis detail.
  assert outs['plan_hypotheses'].shape == (1, ModelConstants.PLAN_MHP_N,
                                           ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH)


@pytest.mark.parametrize("name,shape,single_width,mixture_width", [
  ('plan', ModelConstants.IDX_N * ModelConstants.PLAN_WIDTH, 990, 4955),
  ('lead', ModelConstants.LEAD_MHP_SELECTION * ModelConstants.LEAD_TRAJ_LEN * ModelConstants.LEAD_WIDTH, 144, 102),
])
def test_mixture_detection_keys_off_width(name, shape, single_width, mixture_width):
  parser = Parser()
  assert parser.is_mhp({name: np.zeros((1, single_width), dtype=np.float32)}, name, shape) is False
  assert parser.is_mhp({name: np.zeros((1, mixture_width), dtype=np.float32)}, name, shape) is True


def test_action_and_hidden_state_survive_the_parser():
  # modeld reads action straight off the parsed dict for direct control, and
  # feeds hidden_state back in, so neither may be dropped or reshaped.
  widths = dict(RDF_WIDTHS, action=4, hidden_state=512)
  outs = Parser().parse_outputs(make_outs(widths))

  assert outs['action'].shape == (1, 4)
  assert outs['hidden_state'].shape == (1, 512)
