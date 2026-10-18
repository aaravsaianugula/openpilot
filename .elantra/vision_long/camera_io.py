"""Native NV12 ownership/layout adapter; pixel geometry is preserved without resize."""

from typing import Any

import numpy as np

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
    # Two strided numpy copies (one per plane) instead of a Python row loop: the copy runs in C
    # and releases the GIL, so the card process's controller thread is not held per frame.
    source: np.ndarray = np.frombuffer(data, dtype=np.uint8)
    result: np.ndarray = np.empty((height * 3 // 2, width), dtype=np.uint8)
    result[:height] = source[: stride * height].reshape(height, stride)[:, :width]
    uv_rows = height // 2
    result[height:] = source[uv_offset : uv_offset + stride * uv_rows].reshape(uv_rows, stride)[:, :width]
    return result.tobytes()


def frame_to_bgr(frame: CameraFrame, cv2: Any, np: Any):
    if frame.pixel_format == "nv12":
        raw = np.frombuffer(frame.payload, dtype=np.uint8).reshape(frame.height * 3 // 2, frame.stride)
        return cv2.cvtColor(raw[:, : frame.width], cv2.COLOR_YUV2BGR_NV12)
    if frame.pixel_format == "rgb24":
        raw = np.frombuffer(frame.payload, dtype=np.uint8).reshape(frame.height, frame.stride)
        rgb = raw[:, : frame.width * 3].reshape(frame.height, frame.width, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    raise ValueError("This raw-camera adapter requires NV12 or RGB24")
