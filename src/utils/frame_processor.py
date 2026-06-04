"""
视频帧处理工具。

复用现有 module/video clipping module.py 的核心逻辑，
改为无副作用的函数形式，供 services/sign_service.py 调用。
"""

from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image


def jpeg_to_pil(jpeg_bytes: bytes) -> Image.Image:
    """将 JPEG bytes 转换为 PIL Image（RGB）。"""
    import io
    buf = io.BytesIO(jpeg_bytes)
    return Image.open(buf).convert("RGB")


def resize_frame(frame: cv2.Mat, max_width: int = 640) -> cv2.Mat:
    """等比例缩放帧，不超过 max_width。"""
    h, w = frame.shape[:2]
    if w > max_width:
        ratio = max_width / w
        return cv2.resize(frame, (max_width, int(h * ratio)))
    return frame


def frames_to_video(frame_paths: list[str], output_path: str, fps: int = 2) -> str:
    """将帧序列合成视频（供调试用）。"""
    if not frame_paths:
        raise ValueError("帧列表为空")
    first = cv2.imread(frame_paths[0])
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for p in frame_paths:
        img = cv2.imread(p)
        if img is not None:
            out.write(img)
    out.release()
    return output_path
