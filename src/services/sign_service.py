from __future__ import annotations

import tempfile
from pathlib import Path

import cv2

from src.interfaces import SignLanguageModel
from src.config import CAMERA_INDEX, FRAME_BUFFER_SIZE

# 帧临时保存目录
_FRAME_DIR = Path(tempfile.gettempdir()) / "mute_frames"
_FRAME_DIR.mkdir(exist_ok=True)

# JPEG 编码质量
_JPEG_QUALITY = 55


class SignService:
    """
    手语识别服务。

    职责:
    - 管理摄像头（开/关/读帧）
    - 集成 MediaPipe 手部检测
    - 维护帧缓冲区（用于批量识别）
    - 调用 SignLanguageModel 进行推理
    - 模型未就绪时提供 mock 模式
    """

    def __init__(self, model: SignLanguageModel) -> None:
        self._model = model
        self._camera: cv2.VideoCapture | None = None
        self.buffer: list[dict] = []  # [{jpeg, hands_data, landmarks}]
        self._frame_counter = 0
        self._last_detection: list[dict] = []  # 最近一帧的手部检测数据

    # ------------------------------------------------------------------
    # 摄像头管理
    # ------------------------------------------------------------------

    def start_camera(self) -> bool:
        """打开摄像头。返回是否成功。"""
        if self._camera is not None and self._camera.isOpened():
            return True
        self._camera = cv2.VideoCapture(CAMERA_INDEX)
        if not self._camera.isOpened():
            self._camera = None
            return False
        return True

    def stop_camera(self) -> None:
        """关闭摄像头。"""
        if self._camera is not None:
            self._camera.release()
            self._camera = None

    @property
    def is_capturing(self) -> bool:
        """摄像头是否已打开且可读取。"""
        return self._camera is not None and self._camera.isOpened()

    # ------------------------------------------------------------------
    # 帧读写（带手部检测）
    # ------------------------------------------------------------------

    def read_frame(self) -> dict | None:
        """
        从摄像头读取一帧，运行手部检测，存入缓冲区。

        Returns:
            dict 或 None:
              - jpeg: JPEG bytes（用于前端显示）
              - hands_data: 手部检测数据列表（检测框、关键点等）
              - frame: 原始 BGR numpy 数组（用于后续处理）
        """
        if not self.is_capturing:
            return None
        ret, frame = self._camera.read()
        if not ret:
            return None

        # 镜像翻转（模拟照镜子的效果）
        frame = cv2.flip(frame, 1)

        self._frame_counter += 1

        # MediaPipe 手部检测（如果模型提供）
        hands_data: list[dict] = []
        try:
            if hasattr(self._model, 'detector') and self._model.detector is not None:
                hands_data = self._model.detector.detect(frame)
        except Exception:
            import logging
            logging.getLogger(__name__).debug(
                "MediaPipe 检测异常", exc_info=True,
            )

        # ★ 诊断日志：每30帧输出检测状态
        import logging
        _log = logging.getLogger(__name__)
        if self._frame_counter % 30 == 0:
            if hands_data:
                h0 = hands_data[0]
                _log.info(
                    "📹 帧#%d | 🖐 检测到 %d 只手 | "
                    "hand=%s | conf=%.2f | bbox=%s",
                    self._frame_counter,
                    len(hands_data),
                    h0.get("handedness", "?"),
                    h0.get("confidence", 0),
                    h0.get("bbox", "?"),
                )
            else:
                _log.info(
                    "📹 帧#%d | ❌ 未检测到手部",
                    self._frame_counter,
                )

        # 编码 JPEG
        encode_success, jpeg = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
        )
        if not encode_success:
            return None

        jpeg_bytes = jpeg.tobytes()
        self._last_detection = hands_data

        result = {
            "jpeg": jpeg_bytes,
            "hands_data": hands_data,
            "frame": frame,
            "frame_id": self._frame_counter,
        }

        self.buffer.append(result)

        # 限制缓冲区大小
        if len(self.buffer) > FRAME_BUFFER_SIZE:
            self.buffer.pop(0)

        return result

    def read_frame_simple(self) -> bytes | None:
        """
        简单读帧（不运行检测），返回 JPEG bytes。
        用于兼容不需要检测数据的场景。
        """
        if not self.is_capturing:
            return None
        ret, frame = self._camera.read()
        if not ret:
            return None
        _, jpeg = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
        )
        return jpeg.tobytes()

    @property
    def last_detection(self) -> list[dict]:
        """最近一帧的手部检测数据。"""
        return self._last_detection

    def clear_buffer(self) -> None:
        """清空帧缓冲区。"""
        self.buffer.clear()
        self._last_detection = []
        # 清理临时帧文件
        for f in _FRAME_DIR.iterdir():
            try:
                f.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def _predict_with_fallback(self, frame_paths: list[str]) -> list[str]:
        """
        使用手语模型推理，异常时自动降级为 mock 模式。

        公共方法，被 process_frames() 和 process_frame_files() 复用。
        """
        import logging
        _log = logging.getLogger(__name__)

        try:
            if not self._model.is_loaded():
                _log.info("正在加载手语识别模型（首次推理触发）...")
                self._model.load()
                _log.info("手语识别模型加载完成")

            _log.info("正在推理 %d 帧...", len(frame_paths))
            result = self._model.predict(frame_paths)
            _log.info("推理完成，识别到 %d 个词汇: %s", len(result), result)
            return result
        except Exception:
            _log.warning(
                "真实手语模型推理失败，自动降级为 mock 模式", exc_info=True,
            )
            return self._mock_recognize()

    def process_frames(self) -> list[str]:
        """
        对缓冲区中的帧进行手语识别。

        如果真实模型未实现或加载/推理失败，
        自动降级为 mock 模式，返回预设 Token 列表。
        """
        if not self.buffer:
            return []

        # 将缓冲区 JPEG bytes 写入临时文件
        frame_paths: list[str] = []
        for i, entry in enumerate(self.buffer):
            jpg_bytes = entry["jpeg"] if isinstance(entry, dict) else entry
            path = str(_FRAME_DIR / f"frame_{i:04d}.jpg")
            with open(path, "wb") as f:
                # 处理 dict 和 bytes 两种格式
                if isinstance(jpg_bytes, bytes):
                    f.write(jpg_bytes)
                elif isinstance(jpg_bytes, memoryview):
                    f.write(jpg_bytes.tobytes())
            frame_paths.append(path)

        result = self._predict_with_fallback(frame_paths)
        self.clear_buffer()
        return result

    def process_frame_files(self, frame_paths: list[str]) -> list[str]:
        """
        处理外部帧文件列表（例如从视频文件中抽取的帧）。

        与 process_frames() 共享相同的推理逻辑和 mock 降级策略，
        但不操作摄像头缓冲区。

        Args:
            frame_paths: 已存在的帧图像本地路径列表。

        Returns:
            识别出的手语词汇列表。
        """
        if not frame_paths:
            return []
        return self._predict_with_fallback(frame_paths)

    def _mock_recognize(self) -> list[str]:
        """Mock 模式：返回预设手语 Token，用于流水线联调。"""
        return ["你", "我", "点赞", "我爱你", "讨厌", "打电话", "OK", "摇滚", "胜利", "牛"]
