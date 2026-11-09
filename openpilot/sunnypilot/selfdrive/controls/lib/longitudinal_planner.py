"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import platform

from openpilot.cereal import messaging, custom, log
from opendbc.car import structs
from openpilot.common.constants import CV
from openpilot.common.gps import get_gps_location_service
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD
from openpilot.sunnypilot.selfdrive.controls.lib.e2e_alerts_helper import E2EAlertsHelper
from openpilot.sunnypilot.selfdrive.controls.lib.long_assist import LongAssist, Mode as LongAssistMode
from openpilot.sunnypilot.selfdrive.controls.lib.long_assist_map import EMPTY as MAP_EMPTY, MapContext
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.smart_cruise_control import SmartCruiseControl
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_assist import SpeedLimitAssist
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_resolver import SpeedLimitResolver
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP
from openpilot.sunnypilot.models.helpers import get_active_bundle

LongitudinalPlanSource = custom.LongitudinalPlanSP.LongitudinalPlanSource


class LongitudinalPlannerSP:
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParamsSP, mpc):
    self.events_sp = EventsSP()
    self.resolver = SpeedLimitResolver()
    self.scc = SmartCruiseControl()
    self.resolver = SpeedLimitResolver()
    self.sla = SpeedLimitAssist(CP, CP_SP)
    self.generation = int(model_bundle.generation) if (model_bundle := get_active_bundle()) else None
    self.source = LongitudinalPlanSource.cruise
    self.e2e_alerts_helper = E2EAlertsHelper()
    self.long_assist = LongAssist()
    self.params = Params()
    self.mem_params = Params("/dev/shm/params") if platform.system() != "Darwin" else self.params
    self.gps_location_service = get_gps_location_service(self.params)
    self.map_ctx = MapContext()
    self.frame = 0

    self.output_v_target = 0.
    self.output_a_target = 0.

  def update_targets(self, sm: messaging.SubMaster, v_ego: float, a_ego: float, v_cruise: float) -> tuple[float, float]:
    CS = sm['carState']
    v_cruise_cluster_kph = min(CS.vCruiseCluster, V_CRUISE_MAX)
    v_cruise_cluster = v_cruise_cluster_kph * CV.KPH_TO_MS

    long_enabled = sm['carControl'].enabled
    long_override = sm['carControl'].cruiseControl.override

    # Smart Cruise Control
    self.scc.update(sm, long_enabled, long_override, v_ego, a_ego, v_cruise)

    # Speed Limit Resolver
    self.resolver.update(v_ego, sm)

    # Speed Limit Assist
    has_speed_limit = self.resolver.speed_limit_valid or self.resolver.speed_limit_last_valid
    self.sla.update(long_enabled, long_override, v_ego, a_ego, v_cruise_cluster, self.resolver.speed_limit,
                    self.resolver.speed_limit_final_last, has_speed_limit, self.resolver.distance, self.events_sp)

    targets = {
      LongitudinalPlanSource.cruise: (v_cruise, a_ego),
      LongitudinalPlanSource.sccVision: (self.scc.vision.output_v_target, self.scc.vision.output_a_target),
      LongitudinalPlanSource.sccMap: (self.scc.map.output_v_target, self.scc.map.output_a_target),
      LongitudinalPlanSource.speedLimitAssist: (self.sla.output_v_target, self.sla.output_a_target),
    }

    self.source = min(targets, key=lambda k: targets[k][0])
    self.output_v_target, self.output_a_target = targets[self.source]

    # Long Assist (Elantra overlay)
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self.long_assist.mode = LongAssistMode(min(max(int(self.params.get("LongAssistMode", return_default=True) or 0), 0), 3))
    self.frame += 1
    if self.long_assist.mode == LongAssistMode.off:
      self.long_assist.reset()
    else:
      # test harnesses (longitudinal_maneuvers Plant) pass a plain dict: no validity, no GPS timing
      real_sm = isinstance(sm, messaging.SubMaster)
      map_ahead = MAP_EMPTY
      if real_sm:
        if self.frame % 5 == 0:
          self.map_ctx.set_targets(self.mem_params.get("MapTargetVelocities"))
        gps = sm[self.gps_location_service]
        gps_fresh = 0 < sm.logMonoTime['modelV2'] - sm.logMonoTime[self.gps_location_service] < 2e9
        map_ahead = self.map_ctx.ahead(gps.latitude, gps.longitude, gps.bearingDeg, gps.speed, gps.hasFix and gps_fresh)
      healthy = sm.all_checks(service_list=['modelV2', 'carState', 'radarState']) if real_sm else True
      v_limit = self.resolver.speed_limit_final if self.resolver.speed_limit_valid else 0.
      self.long_assist.wet = sm['selfdriveState'].personality == log.LongitudinalPersonality.standard
      self.long_assist.update(sm['modelV2'], sm['radarState'].leadOne, CS, long_enabled and not long_override,
                              self.output_v_target, v_limit, healthy, map_ahead)

    return self.output_v_target, self.output_a_target

  def update(self, sm: messaging.SubMaster) -> None:
    self.events_sp.clear()
    self.e2e_alerts_helper.update(sm, self.events_sp)

  def publish_longitudinal_plan_sp(self, sm: messaging.SubMaster, pm: messaging.PubMaster) -> None:
    plan_sp_send = messaging.new_message('longitudinalPlanSP')

    plan_sp_send.valid = sm.all_checks(service_list=['carState', 'controlsState'])

    longitudinalPlanSP = plan_sp_send.longitudinalPlanSP
    longitudinalPlanSP.longitudinalPlanSource = self.source
    longitudinalPlanSP.vTarget = float(self.output_v_target)
    longitudinalPlanSP.aTarget = float(self.output_a_target)
    longitudinalPlanSP.events = self.events_sp.to_msg()

    # Smart Cruise Control
    smartCruiseControl = longitudinalPlanSP.smartCruiseControl
    # Vision Control
    sccVision = smartCruiseControl.vision
    sccVision.state = self.scc.vision.state
    sccVision.vTarget = float(self.scc.vision.output_v_target)
    sccVision.aTarget = float(self.scc.vision.output_a_target)
    sccVision.currentLateralAccel = float(self.scc.vision.current_lat_acc)
    sccVision.maxPredictedLateralAccel = float(self.scc.vision.max_pred_lat_acc)
    sccVision.enabled = self.scc.vision.is_enabled
    sccVision.active = self.scc.vision.is_active
    # Map Control
    sccMap = smartCruiseControl.map
    sccMap.state = self.scc.map.state
    sccMap.vTarget = float(self.scc.map.output_v_target)
    sccMap.aTarget = float(self.scc.map.output_a_target)
    sccMap.enabled = self.scc.map.is_enabled
    sccMap.active = self.scc.map.is_active

    # Speed Limit
    speedLimit = longitudinalPlanSP.speedLimit
    resolver = speedLimit.resolver
    resolver.speedLimit = float(self.resolver.speed_limit)
    resolver.speedLimitLast = float(self.resolver.speed_limit_last)
    resolver.speedLimitFinal = float(self.resolver.speed_limit_final)
    resolver.speedLimitFinalLast = float(self.resolver.speed_limit_final_last)
    resolver.speedLimitValid = self.resolver.speed_limit_valid
    resolver.speedLimitLastValid = self.resolver.speed_limit_last_valid
    resolver.speedLimitOffset = float(self.resolver.speed_limit_offset)
    resolver.distToSpeedLimit = float(self.resolver.distance)
    resolver.source = self.resolver.source
    assist = speedLimit.assist
    assist.state = self.sla.state
    assist.enabled = self.sla.is_enabled
    assist.active = self.sla.is_active
    assist.vTarget = float(self.sla.output_v_target)
    assist.aTarget = float(self.sla.output_a_target)

    # E2E Alerts
    e2eAlerts = longitudinalPlanSP.e2eAlerts
    e2eAlerts.greenLightAlert = self.e2e_alerts_helper.green_light_alert
    e2eAlerts.leadDepartAlert = self.e2e_alerts_helper.lead_depart_alert

    # Long Assist
    la = self.long_assist
    longAssist = longitudinalPlanSP.longAssist
    longAssist.mode = int(la.mode)
    longAssist.state = la.state
    longAssist.confidence = float(la.confidence)
    longAssist.confidencePlan = float(la.c_plan)
    longAssist.confidenceLead = float(la.c_lead)
    longAssist.stopDistance = float(la.stop_dist)
    longAssist.brakeActive = la.a_brake is not None
    longAssist.aBrake = float(la.a_brake or 0.)
    longAssist.aLift = float(la.a_lift)
    longAssist.clearStop = la.clear_stop
    longAssist.vCurve = float(la.v_curve)
    longAssist.vTarget = float(la.v_target)
    longAssist.hazardDistance = float(la.hazard_dist)
    longAssist.hazardKind = int(la.hazard_kind)

    pm.send('longitudinalPlanSP', plan_sp_send)
