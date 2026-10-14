"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np

from opendbc.car.structs import car

from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD

RELAXED_MIN_BUCKET_POINTS = np.array([1, 200, 300, 500, 500, 300, 200, 1])

ALLOWED_CARS = ['toyota', 'hyundai', 'rivian', 'honda']


class TorqueEstimatorExt:
  def __init__(self, CP: car.CarParams, *, policy_capture=None):
    self._torqued_policy_capture = policy_capture
    self.CP = CP
    self._params = Params()
    self.frame = -1

    self.enforce_torque_control_toggle = self._params.get_bool("EnforceTorqueControl")  # only during init
    self.use_params = self.CP.brand in ALLOWED_CARS and self.CP.lateralTuning.which() == 'torque'
    if policy_capture is None:
      self.use_live_torque_params = self._params.get_bool("LiveTorqueParamsToggle")
      self.custom_torque_params = self._params.get_bool("CustomTorqueParams")
    else:
      with policy_capture.read("constructorLive", self.frame) as read:
        self.use_live_torque_params = self._params.get_bool("LiveTorqueParamsToggle")
        read.returned(self.use_live_torque_params)
      with policy_capture.read("constructorCustom", self.frame) as read:
        self.custom_torque_params = self._params.get_bool("CustomTorqueParams")
        read.returned(self.custom_torque_params)
    self.torque_override_enabled = self._params.get_bool("TorqueParamsOverrideEnabled")
    self.min_bucket_points = RELAXED_MIN_BUCKET_POINTS
    self.factor_sanity = 0.0
    self.friction_sanity = 0.0
    self.offline_latAccelFactor = 0.0
    self.offline_friction = 0.0

  def initialize_custom_params(self, decimated=False):
    self.update_use_params()

    if self.enforce_torque_control_toggle:
      policy_capture = self._torqued_policy_capture
      if policy_capture is not None:
        with policy_capture.read("initializeRelaxed", self.frame) as read:
          relaxed = self._params.get_bool("LiveTorqueParamsRelaxedToggle")
          read.returned(relaxed)
        if relaxed:
          self.min_bucket_points = RELAXED_MIN_BUCKET_POINTS / (10 if decimated else 1)
          self.factor_sanity = 0.5 if decimated else 1.0
          self.friction_sanity = 0.8 if decimated else 1.0
        with policy_capture.read("initializeCustom", self.frame) as read:
          custom = self._params.get_bool("CustomTorqueParams")
          read.returned(custom)
        if custom:
          self.offline_latAccelFactor = float(self._params.get("TorqueParamsOverrideLatAccelFactor", return_default=True))
          self.offline_friction = float(self._params.get("TorqueParamsOverrideFriction", return_default=True))
        return
      if self._params.get_bool("LiveTorqueParamsRelaxedToggle"):
        self.min_bucket_points = RELAXED_MIN_BUCKET_POINTS / (10 if decimated else 1)
        self.factor_sanity = 0.5 if decimated else 1.0
        self.friction_sanity = 0.8 if decimated else 1.0

      if self._params.get_bool("CustomTorqueParams"):
        self.offline_latAccelFactor = float(self._params.get("TorqueParamsOverrideLatAccelFactor", return_default=True))
        self.offline_friction = float(self._params.get("TorqueParamsOverrideFriction", return_default=True))

  def _update_params(self):
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      policy_capture = self._torqued_policy_capture
      if policy_capture is None:
        self.use_live_torque_params = self._params.get_bool("LiveTorqueParamsToggle")
        self.custom_torque_params = self._params.get_bool("CustomTorqueParams")
      else:
        with policy_capture.read("refreshLive", self.frame) as read:
          self.use_live_torque_params = self._params.get_bool("LiveTorqueParamsToggle")
          read.returned(self.use_live_torque_params)
        with policy_capture.read("refreshCustom", self.frame) as read:
          self.custom_torque_params = self._params.get_bool("CustomTorqueParams")
          read.returned(self.custom_torque_params)
      self.torque_override_enabled = self._params.get_bool("TorqueParamsOverrideEnabled")

  def update_use_params(self):
    self._update_params()

    if self.enforce_torque_control_toggle:
      if self.custom_torque_params and self.torque_override_enabled:
        self.use_params = False
      else:
        self.use_params = self.use_live_torque_params

    self.frame += 1
