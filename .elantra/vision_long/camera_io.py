"""Native NV12 ownership/layout adapter; pixel geometry is preserved without resize."""

from typing import Any

from .transport import CameraFrame


def copy_nv12(data: memoryview, width: int, height: int, stride: int, uv_offset: int) -> bytes:
    """Copy active pixels before VisionIPC reuses its padded shared buffer."""
    if (
        any(type(v) is not int for v in (width, height, stride, uv_offset))
        or not 0 < width <= 8192
        or not 0 < height <= 8192
        or width % 2
        or height % 2
        or not width <= stride <= 65536
        or stride % 2
        or uv_offset < stride * height
    ):
        raise ValueError("Invalid native NV12 descriptor")
    if not isinstance(data, memoryview) or data.ndim != 1 or data.itemsize != 1 or not data.contiguous:
        raise ValueError("Contiguous native byte memoryview required")
    if len(data) < uv_offset + stride * height // 2:
        raise ValueError("Truncated padded native NV12 buffer")
    result = bytearray(width * height * 3 // 2)
    out = 0
    for origin, rows in ((0, height), (uv_offset, height // 2)):
        for row in range(rows):
            offset = origin + row * stride
            result[out : out + width] = data[offset : offset + width]
            out += width
    return bytes(result)


def frame_to_bgr(frame: CameraFrame, cv2: Any, np: Any):
    if frame.pixel_format == "nv12":
        raw = np.frombuffer(frame.payload, dtype=np.uint8).reshape(frame.height * 3 // 2, frame.stride)
        return cv2.cvtColor(raw[:, : frame.width], cv2.COLOR_YUV2BGR_NV12)
    if frame.pixel_format == "rgb24":
        raw = np.frombuffer(frame.payload, dtype=np.uint8).reshape(frame.height, frame.stride)
        rgb = raw[:, : frame.width * 3].reshape(frame.height, frame.width, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    raise ValueError("This raw-camera adapter requires NV12 or RGB24")
