import sys
import types

import numpy as np
import pytest

from cereal import log

import openpilot.selfdrive.modeld.modeld as stock_modeld
import openpilot.sunnypilot.modeld_v2.modeld as modeld_module
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants, Plan
from openpilot.sunnypilot.modeld_v2.modeld import ModelState
from openpilot.sunnypilot.modeld_v2.parse_model_outputs_split import Parser as SplitParser


class DummyFrame:
  def __init__(self, size):
    self.data = bytearray(size)


class FakeOutput:
  def __init__(self, value):
    self.value = np.asarray(value, dtype=np.float32)

  def numpy(self):
    return self.value


class FakeFeatureQueue:
  def __init__(self):
    self.value = np.array([123.0], dtype=np.float32)
    self.numpy_calls = 0
    self.assign_calls = 0
    self.realize_calls = 0

  def numpy(self):
    self.numpy_calls += 1
    return self.value.copy()

  def assign(self, value):
    self.assign_calls += 1
    np.testing.assert_array_equal(value, self.value)
    return self

  def realize(self):
    self.realize_calls += 1
    return self


class PassthroughParser:
  @staticmethod
  def parse_vision_outputs(outputs):
    return outputs

  @staticmethod
  def parse_policy_outputs(outputs):
    return outputs


def _install_tensor_stub(monkeypatch):
  class Tensor:
    @staticmethod
    def from_blob(*_args, **_kwargs):
      return object()

  tinygrad = types.ModuleType('tinygrad')
  tinygrad.__path__ = []
  tensor = types.ModuleType('tinygrad.tensor')
  tensor.Tensor = Tensor
  tinygrad.tensor = tensor
  monkeypatch.setitem(sys.modules, 'tinygrad', tinygrad)
  monkeypatch.setitem(sys.modules, 'tinygrad.tensor', tensor)
  monkeypatch.setattr(modeld_module, 'Tensor', Tensor, raising=False)


def _minimal_state():
  state = ModelState.__new__(ModelState)
  state.frame_buf_params = {'img': (0, 0, 0, 4), 'big_img': (0, 0, 0, 4)}
  state._blob_cache = {}
  state.full_frames = {}
  state.DEV = 'TEST'
  state.WARP_DEV = 'TEST'
  state.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
  state.numpy_inputs = {
    'desire': np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32),
    'traffic_convention': np.zeros((1, 2), dtype=np.float32),
    'tfm': np.zeros((3, 3), dtype=np.float32),
    'big_tfm': np.zeros((3, 3), dtype=np.float32),
  }
  state.input_queues = {}
  state._vision_input_names = ['img', 'big_img']
  state._desire_key = 'desire'
  state._road_key = 'img'
  state._wide_key = 'big_img'
  state._warp_enqueue = lambda **_kwargs: None
  return state


def _run(state, inputs, prepare_only):
  bufs = {name: DummyFrame(state.frame_buf_params[name][3]) for name in state.frame_buf_params}
  transforms = {name: np.eye(3, dtype=np.float32) for name in state.frame_buf_params}
  return state.run(bufs, transforms, inputs, prepare_only)


def _action_state():
  state = ModelState.__new__(ModelState)
  state.LONG_SMOOTH_SECONDS = stock_modeld.LONG_SMOOTH_SECONDS
  state.LAT_SMOOTH_SECONDS = stock_modeld.LAT_SMOOTH_SECONDS
  state.MIN_LAT_CONTROL_SPEED = stock_modeld.MIN_LAT_CONTROL_SPEED
  state.generation = 10
  state.PLANPLUS_CONTROL = 1.0
  state.constants = ModelConstants()
  return state


def _assert_actions_match(actual, expected):
  assert actual.desiredCurvature == pytest.approx(expected.desiredCurvature)
  assert actual.desiredAcceleration == pytest.approx(expected.desiredAcceleration)
  assert actual.shouldStop is expected.shouldStop


def test_action_t_input_receives_lateral_then_longitudinal_delay(monkeypatch):
  _install_tensor_stub(monkeypatch)
  state = _minimal_state()
  state._combined_model_type = 'supercombo'
  state.numpy_inputs['action_t'] = np.zeros((1, 2), dtype=np.float32)
  action_t = np.array([0.18, 0.42], dtype=np.float32)

  result = _run(
    state,
    {
      state.desire_key: np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32),
      'traffic_convention': np.array([1.0, 0.0], dtype=np.float32),
      'action_t': action_t,
    },
    prepare_only=True,
  )

  assert result is None
  np.testing.assert_array_equal(state.numpy_inputs['action_t'], action_t[np.newaxis, :])


def test_legacy_model_without_action_t_still_prepares(monkeypatch):
  _install_tensor_stub(monkeypatch)
  state = _minimal_state()
  state._combined_model_type = 'supercombo'

  result = _run(
    state,
    {
      state.desire_key: np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32),
      'traffic_convention': np.array([1.0, 0.0], dtype=np.float32),
    },
    prepare_only=True,
  )

  assert result is None
  assert 'action_t' not in state.numpy_inputs


def test_run_preserves_off_policy_plan_when_on_policy_has_no_plan_and_round_trips_features(monkeypatch):
  _install_tensor_stub(monkeypatch)
  state = _minimal_state()
  state._combined_model_type = 'multi_policy'
  state.vision_output_slices = {'pose': slice(0, 1)}
  state._policy_keys = ['offPolicy', 'onPolicy']
  state._policy_slices_list = [{'plan': slice(0, 1)}, {'action': slice(0, 1)}]
  state._has_on_policy = True
  state.parser = PassthroughParser()
  feature_queue = FakeFeatureQueue()
  state.input_queues = {'feat_q': feature_queue}
  state._run_policy = lambda **_kwargs: [FakeOutput([1.0]), FakeOutput([42.0]), FakeOutput([7.0])]

  outputs = _run(
    state,
    {
      state.desire_key: np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32),
      'traffic_convention': np.array([1.0, 0.0], dtype=np.float32),
    },
    prepare_only=False,
  )

  np.testing.assert_array_equal(outputs['plan'], [[42.0]])
  np.testing.assert_array_equal(outputs['action'], [[7.0]])
  assert (feature_queue.numpy_calls, feature_queue.assign_calls, feature_queue.realize_calls) == (1, 1, 1)


def test_direct_action_matches_openpilot_lateral_longitudinal_contract():
  prev_action = log.ModelDataV2.Action(desiredCurvature=0.1, desiredAcceleration=-0.2)
  model_output = {'action': np.array([[8.0, 0.6]], dtype=np.float32)}

  actual = ModelState.get_action_from_model(_action_state(), model_output, prev_action, 0.4, 0.6, 20.0)
  expected = stock_modeld.get_action_from_model(model_output, prev_action, 0.4, 0.6, 20.0)

  _assert_actions_match(actual, expected)


def test_split_parser_preserves_direct_action_lateral_longitudinal_order():
  raw = np.array([[3.25, -1.75, 9.0, 11.0]], dtype=np.float32)

  parsed = SplitParser(ignore_missing=True).parse_policy_outputs({'action': raw.copy()})

  np.testing.assert_array_equal(parsed['action'], [[3.25, -1.75]])
  np.testing.assert_allclose(parsed['action_stds'], [[np.exp(9.0), np.exp(11.0)]])


def test_plan_only_action_path_matches_openpilot():
  plan = np.zeros((1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH), dtype=np.float32)
  t_idxs = np.asarray(ModelConstants.T_IDXS, dtype=np.float32)
  plan[0, :, Plan.VELOCITY][:, 0] = 10.0 + 0.4 * t_idxs
  plan[0, :, Plan.ACCELERATION][:, 0] = 0.4
  plan[0, :, Plan.T_FROM_CURRENT_EULER][:, 2] = 0.01 * t_idxs
  plan[0, :, Plan.ORIENTATION_RATE][:, 2] = 0.01
  prev_action = log.ModelDataV2.Action(desiredCurvature=0.02, desiredAcceleration=-0.3)

  actual = ModelState.get_action_from_model(_action_state(), {'plan': plan}, prev_action, 0.4, 0.6, 10.0)
  expected = stock_modeld.get_action_from_model({'plan': plan}, prev_action, 0.4, 0.6, 10.0)

  _assert_actions_match(actual, expected)
