#!/usr/bin/env python3
"""
Round trip of the laptop GPU protocol (openpilot/selfdrive/modeld/laptop.py) against a fake server
that speaks it like .elantra/laptop_gpu/server.py, on a socketpair: no GPU, no cereal.

    python .elantra/test_laptop_gpu.py
"""
import socket
import sys
import threading
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from openpilot.selfdrive.modeld import laptop as link  # noqa: E402

FEED = [["new_img", [1, 12, 128, 256], "uint8"], ["desire", [1, 8], "float32"],
        ["traffic_convention", [1, 2], "float32"], ["action_t", [1, 2], "float32"]]
HELLO = {"ok": True, "model": "fake", "dev": "CUDA", "feed": FEED, "output_slices": {"meta": [0, 4]}, "output_numel": 16}


class FakeSmall:  # the parts of the small ModelState the laptop client reuses: camera copy and warp
  vision_input_names = ("img", "big_img")
  frame_copy_size = 64

  def __init__(self):
    self.frames = np.zeros((2, self.frame_copy_size), dtype=np.uint8)
    self.npy = {"tfm": np.zeros((2, 3, 3), dtype=np.float32)}
    self.input_device = type("Buf", (), {"copy_from": lambda self, src: None})()
    self.input_host, self.warp_inputs, self.warped = None, {}, np.arange(12 * 128 * 256, dtype=np.uint8).reshape(2, 6, 128, 256)

  def run_warp(self):
    return type("T", (), {"numpy": lambda _: self.warped})()


def serve(sock, frames):
  offsets, size = link.layout(FEED)
  buf = bytearray(size)
  for i in range(frames):
    link.recv_into(sock, memoryview(buf))
    seq, = link.FRAME.unpack_from(buf)
    assert seq == i + 1, seq
    new_img = np.frombuffer(buf, np.uint8, 12 * 128 * 256, offsets["new_img"])
    assert np.array_equal(new_img, FakeSmall().warped.reshape(-1)), "warped frames arrive intact"
    desire = np.frombuffer(buf, np.float32, 8, offsets["desire"])
    assert desire[3] == (1. if i == 0 else 0.), f"desire is a rising-edge pulse, frame {i}: {desire}"
    sock.sendall(link.REPLY.pack(seq) + np.full(16, i, np.float32).tobytes())


def main():
  offsets, size = link.layout(FEED)
  assert all(o % link.ALIGN == 0 for o in offsets.values()) and offsets["new_img"] >= link.FRAME.size, offsets
  ends = sorted((o, o + int(np.prod(s)) * np.dtype(d).itemsize) for (_, s, d), o in zip(FEED, offsets.values(), strict=True))
  assert all(a[1] <= b[0] for a, b in zip(ends, ends[1:])) and ends[-1][1] <= size, "inputs do not overlap"

  client, server = socket.socketpair()
  t = threading.Thread(target=serve, args=(server, 3), daemon=True)
  t.start()
  model = link.LaptopModel(FakeSmall(), client, HELLO)
  model.parser = type("P", (), {"parse_outputs": staticmethod(lambda outs: outs)})()  # the real one needs every head
  bufs = {k: type("B", (), {"data": bytes(64)})() for k in ("img", "big_img")}
  desire = np.zeros(8, np.float32)
  desire[3] = 1
  for i in range(3):  # desire held on: the model sees it on the first frame only
    out = model.run(bufs, dict.fromkeys(("img", "big_img"), np.eye(3)), {"desire_pulse": desire.copy(), "traffic_convention": np.array([1., 0.]),
                    "action_t": np.array([.2, .3], np.float32)})
    assert out["meta"].shape == (1, 4) and np.all(out["meta"] == i), out["meta"]
  t.join()

  # the laptop drops out mid-drive: modeld must see an exception, not a hang or a stale frame
  client.settimeout(link.FRAME_TIMEOUT)
  server.close()
  try:
    model.run(bufs, dict.fromkeys(("img", "big_img"), np.eye(3)), {"desire_pulse": desire.copy(), "traffic_convention": np.zeros(2),
              "action_t": np.zeros(2, np.float32)})
    raise AssertionError("a dead link must raise")
  except (ConnectionError, OSError):
    pass

  bad = dict(HELLO, feed=[["new_img", [1, 12, 128, 128], "uint8"]] + FEED[1:])
  try:
    link.LaptopModel(FakeSmall(), socket.socketpair()[0], bad)
    raise AssertionError("a model whose frames differ from the comma warp must be refused")
  except RuntimeError:
    pass
  print("laptop gpu protocol: ok")


if __name__ == "__main__":
  main()
