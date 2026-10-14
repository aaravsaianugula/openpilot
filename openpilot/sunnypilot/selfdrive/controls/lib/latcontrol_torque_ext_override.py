"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from openpilot.common.params import Params


class LatControlTorqueExtOverride:
  def __init__(self, CP):
    self.CP = CP
    self.params = Params()
    policy_capture = getattr(self, "_policy_capture", None)
    if policy_capture is None:
      self.enforce_torque_control_toggle = self.params.get_bool("EnforceTorqueControl")  # only during init
      self.torque_override_enabled = self.params.get_bool("TorqueParamsOverrideEnabled")
    else:
      with policy_capture.read("overrideEnforce") as read:
        self.enforce_torque_control_toggle = self.params.get_bool("EnforceTorqueControl")
        read.returned(self.enforce_torque_control_toggle)
      with policy_capture.read("overrideInitial") as read:
        self.torque_override_enabled = self.params.get_bool("TorqueParamsOverrideEnabled")
        read.returned(self.torque_override_enabled)
    self.frame = -1

  def update_override_torque_params(self, torque_params) -> bool:
    if not self.enforce_torque_control_toggle:
      return False

    self.frame += 1
    if self.frame % 300 == 0:
      policy_capture = getattr(self, "_policy_capture", None)
      if policy_capture is not None:
        with policy_capture.group("overrideRefresh"):
          with policy_capture.read("overrideRefresh") as read:
            self.torque_override_enabled = self.params.get_bool("TorqueParamsOverrideEnabled")
            read.returned(self.torque_override_enabled)
          if not self.torque_override_enabled:
            return False
          with policy_capture.read("overrideFactor") as read:
            value = self.params.get("TorqueParamsOverrideLatAccelFactor", return_default=True)
            read.returned(value)
          torque_params.latAccelFactor = float(value)
          with policy_capture.read("overrideFriction") as read:
            value = self.params.get("TorqueParamsOverrideFriction", return_default=True)
            read.returned(value)
          torque_params.friction = float(value)
          policy_capture.torque_written("override", torque_params)
          return True
      self.torque_override_enabled = self.params.get_bool("TorqueParamsOverrideEnabled")

      if not self.torque_override_enabled:
        return False

      torque_params.latAccelFactor = float(self.params.get("TorqueParamsOverrideLatAccelFactor", return_default=True))
      torque_params.friction = float(self.params.get("TorqueParamsOverrideFriction", return_default=True))
      return True

    return False
