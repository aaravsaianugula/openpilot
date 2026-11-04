import json

import numpy as np

from openpilot.sunnypilot.selfdrive.controls.lib.long_assist_map import EMPTY, Kind, MapContext, DEG, R_EARTH

LAT0, LON0 = 47.36, -122.02
M_LAT = 1 / (DEG * R_EARTH)                       # degrees latitude per metre
M_LON = M_LAT / np.cos(LAT0 * DEG)                # degrees longitude per metre


def north(m, east=0.):
  return LAT0 + m * M_LAT, LON0 + east * M_LON


def write_hazards(tmp_path, pts):
  path = tmp_path / "hazards.npz"
  lat, lon, kind = zip(*[(*north(n, e), k) for n, e, k in pts], strict=True)
  np.savez(path, lat=np.array(lat), lon=np.array(lon), kind=np.array(kind, dtype=np.uint8))
  return str(path)


class TestMapContext:
  def test_invalid_gps_and_missing_file(self, tmp_path):
    ctx = MapContext(str(tmp_path / "none.npz"))
    assert ctx.hazards is None
    assert ctx.ahead(LAT0, LON0, 0., 10., False) is EMPTY

  def test_nearest_hazard_ahead_on_our_line(self, tmp_path):
    ctx = MapContext(write_hazards(tmp_path, [(-40, 0, Kind.signal),   # behind
                                              (120, 0, Kind.stop),
                                              (60, 2, Kind.signal),
                                              (30, 80, Kind.stop),      # off to the side
                                              (400, 0, Kind.stop)]))    # out of range
    ahead = ctx.ahead(LAT0, LON0, 0., 10., True)  # heading north
    assert ahead.hazard_kind == Kind.signal and abs(ahead.hazard_dist - 60.) < 1.
    # heading south the only one ahead is the one that was behind
    ahead = ctx.ahead(LAT0, LON0, 180., 10., True)
    assert ahead.hazard_kind == Kind.signal and abs(ahead.hazard_dist - 40.) < 1.

  def test_bearing_held_when_slow(self, tmp_path):
    ctx = MapContext(write_hazards(tmp_path, [(20, 0, Kind.stop)]))
    ctx.ahead(LAT0, LON0, 0., 10., True)
    assert ctx.ahead(LAT0, LON0, 180., 0.5, True).hazard_kind == Kind.stop  # GPS bearing noise ignored

  def test_curve_targets_along_polyline(self, tmp_path):
    ctx = MapContext(str(tmp_path / "none.npz"))
    pts = [dict(zip(("latitude", "longitude"), north(n), strict=True), velocity=v) for n, v in
           [(-20, 20.), (10, 20.), (60, 15.), (110, 9.)]]
    ctx.set_targets(json.dumps(pts))
    ahead = ctx.ahead(LAT0, LON0, 0., 20., True)
    np.testing.assert_allclose(ahead.curve_dist, [10., 60., 110.], atol=1.)
    np.testing.assert_allclose(ahead.curve_v, [20., 15., 9.])

  def test_curve_targets_for_another_road_ignored(self, tmp_path):
    ctx = MapContext(str(tmp_path / "none.npz"))
    ctx.set_targets(json.dumps([{"latitude": north(500)[0], "longitude": LON0, "velocity": 5.}]))
    assert ctx.ahead(LAT0, LON0, 0., 20., True).curve_dist.size == 0
    ctx.set_targets("not json")
    assert ctx.targets is None
