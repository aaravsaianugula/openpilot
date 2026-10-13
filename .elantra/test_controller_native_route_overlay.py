"""Plural native overlay proof; fixtures never establish vehicle performance."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from controller_model_overlay import RenderError
from controller_native_route import replay_route
from controller_native_route_overlay import validate_plural_arrays

pytest_plugins = ["test_controller_native_route"]


class FixtureController:
  def __init__(self, cp, sp):
    self.long_control_state = 0
    self.pid = SimpleNamespace(i=0.)

  def update(self, active, cs, target, stop, limits):
    self.long_control_state = 0 if not active else 2 if stop else 1
    return -.1 if active and stop else target if active else 0.


def result(episode):
  _, _, static, _, _, rdf, tape = episode
  output = replay_route(tape, static[0], FixtureController, lambda *args: (-3.5, 2.))
  return output, tape, rdf, float(tape["raw_censor_t"])


def test_plural_native_validation_keeps_actual_censor_and_cold_flags(episode):
  output, tape, rdf, censor = result(episode)
  counts = validate_plural_arrays(output, tape, rdf, censor)
  assert counts["native_admitted"] == int(output["admitted"].sum())
  assert not output["controlled_command_available"][output["t"] >= censor].any()


@pytest.mark.parametrize("mutation", ["raw", "future_index", "target", "stop", "censor", "controlled", "cold", "bounds", "admission"])
def test_plural_native_overlay_rejects_altered_output_proof(episode, mutation):
  output, tape, rdf, censor = result(episode)
  output = deepcopy(output)
  tick = np.flatnonzero(output["admitted"])[0]
  if mutation == "raw":
    output["a"][tick] += .1
  elif mutation == "future_index":
    output["rdf_index"][tick] += 1
  elif mutation == "target":
    output["rdf_accel_target"][tick] += .1
  elif mutation == "stop":
    output["should_stop"][tick] = False
  elif mutation == "censor":
    output["raw_censor_t"] += .1
  elif mutation == "controlled":
    output["controlled_command_available"][output["t"] >= censor] = True
  elif mutation == "cold":
    output["initialization_uncertain"][tick] = False
  elif mutation == "bounds":
    output["native_accel"][tick] = 2.1
  else:
    output["admitted"][tick] = False
  with pytest.raises(RenderError):
    validate_plural_arrays(output, tape, rdf, censor)
