"""
A laptop GPU in place of a chestnut.

The comma's aux USB-C port runs a CDC-NCM gadget (192.168.250.1/24); a laptop plugged into it
gets a USB network adapter (192.168.250.2) and runs .elantra/laptop_gpu/server.py, which holds the
chestnut-class model and its recurrent state on its own GPU. Each frame the comma warps both
cameras on its own GPU exactly as for the small model and sends only the warped frames plus the
float inputs (~400 KB); the laptop answers with the model's output vector.

Top-level imports are stdlib + numpy only: the laptop server imports the protocol from here.
"""
import json
import os
import socket
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

HOST, PORT = "192.168.250.2", 8390
FRAME = struct.Struct("<I")  # seq: client -> server, then each feed input at its layout() offset
REPLY = struct.Struct("<I")  # seq: server -> client, followed by the float32 outputs
FRAME_TIMEOUT = 0.1  # a late frame drops to the small model, like a chestnut fault
# comma's newest chestnut model published as ONNX (later ones ship as AMD binaries only)
DEFAULT_ONNX = ("6fee5937923c74848df4a63f6239eb6331c6274dd4bdb7a5d6ec0388a8b543d5", 766018462, "Cinque Terre v2 (Sep 14, 2026)")
FEED = ("new_img", "desire", "traffic_convention", "action_t")  # what this modeld fills each frame
ALIGN = 128  # the laptop GPU reads each input in place, so each starts aligned

UDC = Path("/sys/class/udc/a600000.dwc3")
GADGET = Path("/sys/kernel/config/usb_gadget/laptop")
GADGET_UP = f"""set -e
g={GADGET}
if [ "$(cat $g/UDC 2>/dev/null)" != "{UDC.name}" ]; then
  mkdir -p $g/strings/0x409 $g/configs/c.1/strings/0x409 $g/functions/ncm.usb0
  echo 0x1d6b > $g/idVendor; echo 0x0104 > $g/idProduct; echo 0x0200 > $g/bcdUSB
  echo 0xef > $g/bDeviceClass; echo 0x02 > $g/bDeviceSubClass; echo 0x01 > $g/bDeviceProtocol
  echo comma.ai > $g/strings/0x409/manufacturer; echo "laptop gpu link" > $g/strings/0x409/product
  echo laptop-gpu > $g/strings/0x409/serialnumber; echo NCM > $g/configs/c.1/strings/0x409/configuration
  [ -e $g/configs/c.1/ncm.usb0 ] || ln -s $g/functions/ncm.usb0 $g/configs/c.1/
  echo {UDC.name} > $g/UDC
fi
i=$(cat $g/functions/ncm.usb0/ifname)
! command -v nmcli >/dev/null || nmcli device set $i managed no || true
ip addr replace 192.168.250.1/24 dev $i
ip link set $i up
"""
_next_bring_up = 0.


def send_msg(sock: socket.socket, obj) -> None:
  data = json.dumps(obj).encode()
  sock.sendall(struct.pack("<I", len(data)) + data)


def recv_into(sock: socket.socket, view: memoryview) -> None:
  while view:
    n = sock.recv_into(view)
    if not n:
      raise ConnectionError("laptop closed the link")
    view = view[n:]


def recv_msg(sock: socket.socket):
  size = bytearray(4)
  recv_into(sock, memoryview(size))
  data = bytearray(struct.unpack("<I", size)[0])
  recv_into(sock, memoryview(data))
  return json.loads(data)


def layout(feed) -> tuple[dict[str, int], int]:
  """Byte offset of each feed input in a frame message, and the message size."""
  offsets, end = {}, ALIGN
  for name, shape, dtype in feed:
    offsets[name] = end
    end += -(-int(np.prod(shape)) * np.dtype(dtype).itemsize // ALIGN) * ALIGN
  return offsets, end


def link_up() -> bool:
  """True when a laptop has enumerated the aux-port network gadget."""
  try:
    ifname = (GADGET / "functions/ncm.usb0/ifname").read_text().strip()
    return Path(f"/sys/class/net/{ifname}/operstate").read_text().strip() == "up"
  except OSError:
    return False


def ensure_link() -> bool:
  """link_up(), bringing the gadget up first if needed. Only hardwared calls this."""
  global _next_bring_up
  try:
    bound = (GADGET / "UDC").read_text().strip() == UDC.name
  except OSError:
    bound = False
  if UDC.exists() and not bound and time.monotonic() > _next_bring_up:
    _next_bring_up = time.monotonic() + 30
    try:
      subprocess.run(["sudo", "sh", "-c", GADGET_UP], capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
      pass
  return link_up()


class LaptopModel:
  chestnut = True  # modeld treats it as the big model: modelV2.big, and any failure falls back to the small model

  def __init__(self, small, sock: socket.socket, hello: dict):
    from openpilot.selfdrive.modeld.constants import ModelConstants
    from openpilot.selfdrive.modeld.parse_model_outputs import Parser
    self.small, self.sock = small, sock
    self.vision_input_names = small.vision_input_names
    self.output_slices = {k: slice(*v) for k, v in hello["output_slices"].items()}
    self.output_bytes = hello["output_numel"] * 4
    self.parser = Parser()
    self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    self.seq = 0
    offsets, size = layout(hello["feed"])
    self.tx = bytearray(size)
    self.slots = {name: np.ndarray(shape, dtype=dtype, buffer=self.tx, offset=offsets[name]) for name, shape, dtype in hello["feed"]}
    self.rx = bytearray(REPLY.size)
    warped = small.run_warp(**small.warp_inputs).numpy()
    if warped.size != self.slots['new_img'].size or warped.dtype != self.slots['new_img'].dtype:
      raise RuntimeError(f"comma warp gives {warped.dtype}{warped.shape}, laptop model takes {self.slots['new_img'].dtype}{self.slots['new_img'].shape}")

  def run(self, bufs, transforms, inputs, after_enqueue=None) -> dict[str, np.ndarray]:
    s = self.small
    for i, key in enumerate(self.vision_input_names):
      np.copyto(s.frames[i], np.frombuffer(bufs[key].data, dtype=np.uint8, count=s.frame_copy_size))
      s.npy['tfm'][i] = transforms[key]
    s.input_device.copy_from(s.input_host)
    self.slots['new_img'][...] = s.run_warp(**s.warp_inputs).numpy().reshape(self.slots['new_img'].shape)

    # same desire pulse as ModelState.run: the model decides when an action completes
    inputs['desire_pulse'][0] = 0
    self.slots['desire'][...] = np.where(inputs['desire_pulse'] - self.prev_desire > .99, inputs['desire_pulse'], 0)
    self.prev_desire[:] = inputs['desire_pulse']
    self.slots['traffic_convention'][...] = inputs['traffic_convention']
    self.slots['action_t'][...] = inputs['action_t']

    self.seq += 1
    FRAME.pack_into(self.tx, 0, self.seq)
    self.sock.sendall(self.tx)
    if after_enqueue is not None:
      after_enqueue()
    recv_into(self.sock, memoryview(self.rx))
    seq, = REPLY.unpack(self.rx)
    model_output = np.empty(self.output_bytes // 4, dtype=np.float32)
    recv_into(self.sock, memoryview(model_output).cast('B'))
    if seq != self.seq or not np.all(np.isfinite(model_output)):
      raise RuntimeError(f"laptop sent a bad frame (seq {seq} != {self.seq} or not finite)")
    outputs = self.parser.parse_outputs({k: model_output[np.newaxis, v] for k, v in self.output_slices.items()})
    if os.getenv('SEND_RAW_PRED'):
      outputs['raw_pred'] = model_output.copy()
    return outputs


def load(small, params):
  """The laptop's model for this drive, or None to keep the small one."""
  from openpilot.common.swaglog import cloudlog
  bundle = params.get("ModelManager_ActiveBundleChestnut") or {}
  params.put_bool("ChestnutLoading", True)
  model = None
  try:
    sock = socket.create_connection((HOST, PORT), timeout=5)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    send_msg(sock, {"ref": bundle.get("ref"), "name": bundle.get("display_name")})
    sock.settimeout(60)  # loading a cached model onto the laptop GPU
    hello = recv_msg(sock)
    if not hello.get("ok"):
      raise RuntimeError(hello.get("error"))
    if sorted(name for name, _, _ in hello["feed"]) != sorted(FEED):
      raise RuntimeError(f"laptop model takes {[n for n, _, _ in hello['feed']]}, this modeld fills {FEED}")
    sock.settimeout(FRAME_TIMEOUT)
    model = LaptopModel(small, sock, hello)
    cloudlog.warning(f"laptop gpu: {hello['model']} on {hello['dev']}")
  except Exception:
    cloudlog.exception("laptop gpu unavailable, small model drives")
  params.put_bool("ChestnutActive", model is not None)
  return model
