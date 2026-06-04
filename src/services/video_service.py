"""
视频处理服务。

封装视频抽帧、网络视频下载逻辑。
从 module/video clipping module.py 重构为服务形式。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import requests
from PIL import Image


class VideoService:
    """
    视频处理服务。

    职责：
    - 下载网络视频到本地临时文件
    - 从视频中按固定间隔抽取帧（手语专用：每 0.8 秒一帧）
    - 管理临时帧文件目录
    """

    def __init__(self, frame_interval_sec: float = 0.8) -> None:
        self.frame_interval_sec = frame_interval_sec
        self._frame_dir = Path(tempfile.gettempdir()) / "mute_video_frames"
        self._frame_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 网络视频下载
    # ------------------------------------------------------------------

    def download_if_url(self, input_path: str) -> str:
        """
        如果是网络 URL 则下载到本地临时文件；本地路径直接返回。

        支持图片（jpg/png）和视频（mp4/mov/avi/mkv/flv）。

        Args:
            input_path: 本地路径 或 HTTP/HTTPS URL。

        Returns:
            本地临时文件路径。
        """
        if not input_path.startswith(("http://", "https://")):
            return input_path

        # 自动识别文件后缀
        suffix = ".mp4"
        lower = input_path.lower()
        if lower.endswith((".jpg", ".jpeg", ".png", ".bmp")):
            suffix = ".jpg"
        elif lower.endswith((".mov", ".avi", ".mkv", ".flv")):
            suffix = ".mp4"

        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_path = temp_file.name

        try:
            resp = requests.get(input_path, stream=True, timeout=30)
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    temp_file.write(chunk)
            temp_file.close()
            return temp_path
        except Exception:
            temp_file.close()
            Path(temp_path).unlink(missing_ok=True)
            raise

    # ------------------------------------------------------------------
    # 视频抽帧
    # ------------------------------------------------------------------

    def extract_frames(self, video_path: str, num_frames: int | None = None) -> list[str]:
        """
        从视频文件中按间隔抽取帧。

        手语专用：每 0.8 秒抽一帧。

        Args:
            video_path: 视频文件本地路径。
            num_frames: 可选，限制最大抽取帧数。

        Returns:
            帧图像本地路径列表。
        """
        cap = cv2.VideoCapture(video_path)
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25  # 兜底

            frame_interval = int(fps * self.frame_interval_sec)
            if frame_interval < 1:
                frame_interval = 1

            indices = list(range(0, total, frame_interval))
            if num_frames and len(indices) > num_frames:
                indices = indices[:num_frames]

            frames: list[str] = []
            for i, idx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                save_path = str(self._frame_dir / f"vid_frame_{i:04d}.jpg")
                img.save(save_path)
                frames.append(save_path)

            return frames
        finally:
            cap.release()

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------

    def clear_frames(self) -> None:
        """清空临时帧目录中所有文件。"""
        for f in self._frame_dir.iterdir():
            try:
                f.unlink()
            except OSError:
                pass

    @property
    def frame_dir(self) -> Path:
        return self._frame_dir
