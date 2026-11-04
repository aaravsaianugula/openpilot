import long_assist_replay as lar


def f(t, v, brk, a_brake=0.):
  return {"t": t, "v": v, "brk": brk, "brake_active": a_brake < 0., "a_brake": a_brake, "state": "stop",
          "stop_dist": 40., "lead_d": -1.}


def test_human_stop_found_and_la_lead_time():
  # cruise at 10 m/s, Long Assist brakes from t=1, driver from t=2, standstill at t=4
  frames = [f(i * 0.5, 10., False, -1.5 if i >= 2 else 0.) for i in range(4)]
  frames += [f(2. + i * 0.5, 10. - 2.5 * i, True, -1.5) for i in range(5)]
  frames += [f(4.5, 0., True)]
  (hs,) = lar.human_stops(frames)
  assert hs["t"] == 2. and hs["v"] == 10. and hs["la_lead_s"] == 1.
  assert hs["dist"] == 12.5 and hs["a_driver"] == -4.


def test_slow_creep_stop_is_not_counted():
  frames = [f(i * 0.5, 3. - i, True) for i in range(4)] + [f(2., 0., True)]
  assert lar.human_stops(frames) == []
