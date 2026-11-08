# NVIDIA driver profile (global): "Power management mode" = prefer maximum performance, through the driver's own nvapi64.dll.
# usage: nvdrs.py apply | restore (back to the driver default) | show
import ctypes as C, sys

PMM = 0x1057EB71  # "Power management mode"; 1 = prefer maximum performance

nv = C.WinDLL("nvapi64.dll")
nv.nvapi_QueryInterface.restype = C.c_void_p
nv.nvapi_QueryInterface.argtypes = [C.c_uint32]
def fn(fid, *args):
  p = nv.nvapi_QueryInterface(fid)
  if not p:
    raise OSError(f"nvapi function {fid:#x} missing")
  return C.CFUNCTYPE(C.c_int, *args)(p)

class Setting(C.Structure):  # NVDRS_SETTING_V1, 12320 bytes
  _fields_ = [("version", C.c_uint32), ("name", C.c_wchar * 2048), ("id", C.c_uint32), ("type", C.c_uint32),
              ("location", C.c_uint32), ("isCurrentPredefined", C.c_uint32), ("isPredefinedValid", C.c_uint32),
              ("predefined", C.c_uint32), ("_p", C.c_uint8 * 4096), ("current", C.c_uint32), ("_c", C.c_uint8 * 4096)]
assert C.sizeof(Setting) == 12320, C.sizeof(Setting)
VER = C.sizeof(Setting) | (1 << 16)

def ok(r, what, allow=()):
  if r != 0 and r not in allow:
    raise OSError(f"{what} failed: nvapi status {r}")

cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
if cmd not in ("apply", "restore", "show"):
  raise SystemExit(__doc__)
H = C.c_void_p
ok(fn(0x0150E828)(), "Initialize")
sess, prof = H(), H()
ok(fn(0x0694D52E, C.POINTER(H))(C.byref(sess)), "CreateSession")
ok(fn(0x375DBD6B, H)(sess), "LoadSettings")
ok(fn(0xDA8466A0, H, C.POINTER(H))(sess, C.byref(prof)), "GetBaseProfile")

s = Setting(version=VER)
r = fn(0x73BF8338, H, H, C.c_uint32, C.POINTER(Setting))(sess, prof, PMM, C.byref(s))
print("Power management mode:", "default (unset)" if r == -160 else (s.current if r == 0 else f"error {r}"))
if cmd == "apply":
  ok(fn(0x577DD202, H, H, C.POINTER(Setting))(sess, prof, C.byref(Setting(version=VER, id=PMM, type=0, current=1))), "SetSetting")  # type 0 = DWORD
elif cmd == "restore":
  ok(fn(0xE4A26362, H, H, C.c_uint32)(sess, prof, PMM), "DeleteProfileSetting", allow=(-160,))  # -160: already default
if cmd != "show":
  ok(fn(0xFCBC7E14, H)(sess), "SaveSettings")
  print(cmd, "done")
fn(0xDAD9CFF8, H)(sess)
