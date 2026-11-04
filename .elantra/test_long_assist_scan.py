#!/usr/bin/env python3
"""Tests for long_assist_scan.py. No logs: pure functions over synthetic frames."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import long_assist_scan as las


def frame(t, plan, a_brake=0., brake_active=False, lift=0.):
  return {"t": t, "v": 10., "plan": plan, "mode": 1, "state": "stop", "conf": 0.8,
          "brake_active": brake_active, "a_brake": a_brake, "lift_v": lift, "stop_dist": 30.}


def test_brake_episode_found_and_short_noise_dropped():
  frames = [frame(i * 0.05, -0.3, -2.0, True) for i in range(10)]
  frames += [frame(1. + i * 0.05, -0.3) for i in range(5)]
  frames += [frame(2. + i * 0.05, -0.3, -2.0, True) for i in range(las.MIN_FRAMES - 1)]
  rep = las.summarize(frames)
  assert len(rep["brake_episodes"]) == 1
  ep = rep["brake_episodes"][0]
  assert ep["t"] == 0. and ep["worst_gap"] == 1.7
  assert rep["lift_episodes"] == []


def test_brake_matching_planner_is_not_an_episode():
  rep = las.summarize([frame(i * 0.05, -2.0, -2.1, True) for i in range(10)])
  assert rep["brake_episodes"] == []


def test_lift_episode():
  rep = las.summarize([frame(i * 0.05, 0.2, lift=1.0) for i in range(8)])
  assert len(rep["lift_episodes"]) == 1 and rep["conf_on_share"] == 1.0


if __name__ == "__main__":
  import pytest
  sys.exit(pytest.main([__file__, "-q"]))
