"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.common.params import Params


def get_lat_delay(params: Params, stock_lat_delay: float, *, policy_capture=None) -> float:
# live learning on: use what lagd publishes.
# off: use the fixed steerActuatorDelay + software delay sum that LagdToggle caches.

  if policy_capture is None:
    if params.get_bool("LagdToggle"):
      return stock_lat_delay
    return float(params.get("LagdValueCache", return_default=True))
  with policy_capture.read("lagdToggle") as read:
    toggle = params.get_bool("LagdToggle")
    read.returned(toggle)
  if toggle:
    policy_capture.delay_origin = "publishedDelay"
    return stock_lat_delay
  with policy_capture.read("delayCache") as read:
    value = params.get("LagdValueCache", return_default=True)
    read.returned(value)
  policy_capture.delay_origin = "cachedDelay"
  return float(value)
