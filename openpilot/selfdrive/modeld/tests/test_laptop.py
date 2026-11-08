import socket
import threading
import time

import numpy as np
import pytest

from openpilot.selfdrive.modeld import laptop
from openpilot.selfdrive.modeld.laptop import Watchdog, LaptopModel, REPLY, FRAME_TIMEOUT


def feed_dts(wd, dts):
  for dt in dts:
    wd.frame(dt)


class TestWatchdog:
  def test_fast_frames(self):
    wd = Watchdog()
    feed_dts(wd, [0.02] * 60)
    assert not wd.lagging

  def test_sustained_lag_warns_without_raising(self):
    wd = Watchdog()
    feed_dts(wd, [0.047] * 60)
    assert wd.lagging
    feed_dts(wd, [0.02] * 20)
    assert not wd.lagging

  def test_three_slow_frames_raise(self):
    wd = Watchdog()
    feed_dts(wd, [0.06, 0.06, 0.02, 0.06, 0.06])  # not consecutive
    with pytest.raises(RuntimeError):
      feed_dts(wd, [0.06])


class FakeBuf:
  data = bytes(16)


class FakeSmall:
  vision_input_names = ["img"]
  frame_copy_size = 16

  def __init__(self):
    self.frames = [np.zeros(16, dtype=np.uint8)]
    self.npy = {'tfm': [None]}
    self.input_device = self
    self.input_host = None
    self.warp_inputs = {}

  def copy_from(self, _):
    pass

  def run_warp(self, **_):
    return self

  def numpy(self):
    return np.zeros(4, dtype=np.uint8)


def make_model(server_fn):
  """LaptopModel over a socketpair; server_fn(sock, model) plays the laptop."""
  a, b = socket.socketpair()
  a.settimeout(FRAME_TIMEOUT)
  hello = {"output_slices": {"x": [0, 2]}, "output_numel": 2, "feed": [
    ["new_img", [4], "uint8"], ["desire", [8], "float32"], ["traffic_convention", [2], "float32"], ["action_t", [2], "float32"]]}
  m = LaptopModel(FakeSmall(), a, hello)
  m.parser.parse_outputs = lambda o: o
  threading.Thread(target=server_fn, args=(b, m), daemon=True).start()
  return m


def run(m):
  inputs = {'desire_pulse': np.zeros(m.prev_desire.size, dtype=np.float32),
            'traffic_convention': np.zeros(2, np.float32), 'action_t': np.zeros(2, np.float32)}
  return m.run({"img": FakeBuf()}, {"img": np.eye(3)}, inputs)


def serve(delay=0., seq_off=0, close_after=None):
  def fn(sock, m):
    n = 0
    while close_after is None or n < close_after:
      try:
        sock.recv_into(bytearray(len(m.tx)), len(m.tx), socket.MSG_WAITALL)
      except OSError:
        return
      n += 1
      time.sleep(delay)
      sock.sendall(REPLY.pack(n + seq_off) + np.ones(2, np.float32).tobytes())
    sock.close()
  return fn


class TestLaptopModel:
  def test_ok(self):
    m = make_model(serve())
    for _ in range(5):
      assert run(m)["x"].shape == (1, 2)
    assert not m.lagging

  def test_sustained_lag_flags_but_runs(self):
    m = make_model(serve(delay=0.047))
    for _ in range(25):
      run(m)
    assert m.lagging

  def test_three_slow_frames_raise(self):
    m = make_model(serve(delay=0.055))
    with pytest.raises(RuntimeError):
      for _ in range(3):
        run(m)

  def test_timeout_raises(self):
    m = make_model(serve(delay=0.3))
    with pytest.raises(TimeoutError):
      run(m)

  def test_disconnect_raises(self):
    m = make_model(serve(close_after=1))
    run(m)
    with pytest.raises(OSError):
      run(m)

  def test_bad_seq_raises(self):
    m = make_model(serve(seq_off=5))
    with pytest.raises(RuntimeError):
      run(m)
