import json

import numpy as np

from openpilot.sunnypilot.selfdrive.controls.lib.long_assist_map import EMPTY, Cert, Kind, MapContext, DEG, R_EARTH

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
                                              (60, 15, Kind.signal),    # junction-centre signal, inside its wide corridor
                                              (30, 12, Kind.stop),      # side-street stop sign along an arterial
                                              (30, 80, Kind.signal),    # off to the side
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


AP_N = 8


def write_approaches(tmp_path, hazards):
  """hazards: (north, east, kind, [(points back from the node as (north, east) metres, stop line offset, cert, half-width)])"""
  rows = sorted(hazards, key=lambda h: h[0])
  lat, lon, kind, i0, pts, n, off, cert, w = [], [], [], [0], [], [], [], [], []
  for nn, ee, k, aps in rows:
    la, lo = north(nn, ee)
    lat.append(la)
    lon.append(lo)
    kind.append(k)
    for poly, o, c, hw in aps:
      p = np.array([(e - ee, q - nn) for q, e in poly], dtype=np.float32)
      n.append(len(p))
      pts.append(np.concatenate([p, np.repeat(p[-1:], AP_N - len(p), axis=0)]))
      off.append(o)
      cert.append(c)
      w.append(hw)
    i0.append(len(n))
  path = tmp_path / "hazards_ap.npz"
  np.savez(path, lat=np.array(lat), lon=np.array(lon), kind=np.array(kind, np.uint8), ap_i0=np.array(i0, np.int32),
           ap_pts=np.array(pts, np.float32).reshape(-1, AP_N, 2), ap_n=np.array(n, np.uint8), ap_off=np.array(off, np.float32),
           ap_cert=np.array(cert, np.uint8), ap_w=np.array(w, np.float32))
  return str(path)


def straight(n0, e0, dn, de, length=250.):
  """from the node (n0, e0) back along (dn, de)"""
  return [(n0 + dn * s, e0 + de * s) for s in (0., 30., 100., length)]


class TestApproaches:
  def test_our_approach_only(self, tmp_path):
    ctx = MapContext(write_approaches(tmp_path, [
      (80., 0., Kind.stop, [(straight(80., 0., -1., 0.), 0., Cert.sure, 4.)]),         # ours: entered from the south
      (40., 8., Kind.stop, [(straight(40., 8., 0., 1.), 0., Cert.sure, 4.)]),          # side street, entered from the east
      (60., 0., Kind.stop, [(straight(60., 0., 1., 0.), 0., Cert.sure, 4.)]),          # the other direction's sign
      (100., 30., Kind.signal, [(straight(100., 30., -1., 0.), 20., Cert.sure, 8.)]),  # parallel road 30 m east
    ]))
    ahead = ctx.ahead(LAT0, LON0, 0., 10., True, 1.)
    assert ahead.hazard_kind == Kind.stop and ahead.hazard_cert == Cert.sure and abs(ahead.hazard_dist - 80.) < 1.
    assert ctx.ahead(LAT0, LON0, 180., 10., True, 1.).hazard_dist < 0.  # heading south: none of them is ours

  def test_junction_centre_signal_stop_line_and_lane_offset(self, tmp_path):
    ctx = MapContext(write_approaches(tmp_path, [(150., 0., Kind.signal, [(straight(150., 0., -1., 0.), 22., Cert.sure, 10.)])]))
    lat, lon = north(0., 6.)  # in the right lane of a wide road, 6 m off its centreline
    ahead = ctx.ahead(lat, lon, 0., 15., True, 2.)
    assert ahead.hazard_kind == Kind.signal and abs(ahead.hazard_dist - 128.) < 1.

  def test_curved_approach_followed(self, tmp_path):
    # the road bends: the node is 60 m east of our heading line but we are on its approach
    poly = [(200., 60.), (150., 50.), (100., 25.), (50., 8.), (0., 0.), (-60., 0.)]
    ctx = MapContext(write_approaches(tmp_path, [(200., 60., Kind.stop, [(poly, 0., Cert.sure, 4.)])]))
    ahead = ctx.ahead(LAT0, LON0, 5., 10., True, 1.)
    assert ahead.hazard_kind == Kind.stop and 205. < ahead.hazard_dist < 215.

  def test_bad_fix_matches_nothing(self, tmp_path):
    ctx = MapContext(write_approaches(tmp_path, [(80., 0., Kind.stop, [(straight(80., 0., -1., 0.), 0., Cert.sure, 4.)])]))
    assert ctx.ahead(LAT0, LON0, 0., 10., True, 25.) is EMPTY

  def test_heading_follows_yaw_when_slow(self, tmp_path):
    # stopped facing east, then turning left onto the road north at walking pace: GPS bearing is noise, the yaw rate is not
    ctx = MapContext(write_approaches(tmp_path, [(80., 0., Kind.stop, [(straight(80., 0., -1., 0.), 0., Cert.sure, 4.)])]))
    ctx.ahead(LAT0, LON0, 90., 10., True, 1.)
    for _ in range(40):  # -90 deg over 2 s (clockwise +)
      ahead = ctx.ahead(LAT0, LON0, 200., 1., True, 1., -np.pi / 4, 0.05)
    assert min(abs(ctx.bearing), abs(ctx.bearing - 360.)) < 1.
    assert ahead.hazard_kind == Kind.stop
