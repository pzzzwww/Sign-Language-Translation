from __future__ import annotations

import logging

from src.interfaces import SignLanguageModel


class SignService:
    """
    手语识别服务。

    职责:
    - 持有 SignLanguageModel 实例
    - 对外提供帧文件识别入口（视频上传路径）
    - 推理异常时自动降级为 mock 模式
    """

    def __init__(self, model: SignLanguageModel) -> None:
        self._model = model

    @property
    def model(self) -> SignLanguageModel:
        """当前持有的手语识别模型（供实时流复用检测器/识别器）。"""
        return self._model

    def reset_session(self) -> None:
        """重置模型会话状态（识别器缓冲区、检测器跟踪状态等）。"""
        if hasattr(self._model, "reset_session"):
            self._model.reset_session()

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def process_frame_files(self, frame_paths: list[str]) -> list[str]:
        """
        处理外部帧文件列表（例如从视频文件中抽取的帧）。

        Args:
            frame_paths: 已存在的帧图像本地路径列表。

        Returns:
            识别出的手语词汇列表。
        """
        if not frame_paths:
            return []
        return self._predict_with_fallback(frame_paths)

    def _predict_with_fallback(self, frame_paths: list[str]) -> list[str]:
        """
        使用手语模型推理，异常时自动降级为 mock 模式。
        """
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

    def _mock_recognize(self) -> list[str]:
        """Mock 模式：返回预设手语 Token，用于流水线联调。"""
        return ["你", "我", "点赞", "我爱你", "讨厌", "打电话", "OK", "摇滚", "胜利", "牛"]
