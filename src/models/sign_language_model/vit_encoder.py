"""
ViT 视觉特征编码器 — 对手部检测区域提取 ViT-B/16 视觉特征。

与 MediaPipe 关键点形成多模态输入，提升手势识别鲁棒性。

Usage:
    encoder = ViTFeatureExtractor()
    encoder.load()
    features = encoder.extract(frame, bbox)  # → (768,) float32
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import ViTImageProcessor, ViTModel
    _VIT_AVAILABLE = True
except ImportError:
    _VIT_AVAILABLE = False
    torch = None  # type: ignore
    ViTModel = None
    ViTImageProcessor = None


class ViTFeatureExtractor:
    """
    基于 HuggingFace ViT-B/16 的手部区域视觉特征提取器。

    冻结 backbone，仅推理模式，提取 CLS token 作为全局视觉特征。
    输入 224×224 RGB 手部裁剪，输出 768 维特征向量。

    特性:
      - 懒加载，首次 extract() 时自动初始化
      - 支持 CPU 推理（~100ms/帧 on modern CPU）
      - 无效输入（无手部）返回零向量
    """

    VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"
    CROP_SIZE = 224

    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._model: Optional[ViTModel] = None
        self._processor: Optional[ViTImageProcessor] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # 加载 / 释放
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self._loaded:
            return

        if not _VIT_AVAILABLE:
            logger.warning(
                "transformers 未安装，ViT 特征提取不可用。"
                "安装方式: pip install transformers"
            )
            self._loaded = True
            return

        logger.info("加载 ViT-B/16 视觉编码器: %s", self.VIT_MODEL_NAME)
        self._processor = ViTImageProcessor.from_pretrained(self.VIT_MODEL_NAME)
        self._model = ViTModel.from_pretrained(self.VIT_MODEL_NAME).to(self._device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False

        self._loaded = True
        logger.info("ViT-B/16 就绪（冻结 backbone，%s）", self._device)

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def feature_dim(self) -> int:
        """输出特征向量维度（ViT-B/16 hidden_size = 768）。"""
        return 768

    # ------------------------------------------------------------------
    # 特征提取
    # ------------------------------------------------------------------

    def extract(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        从帧中裁剪手部区域并提取 ViT 特征。

        Args:
            frame: (H, W, 3) BGR uint8 numpy array。
            bbox: (x, y, w, h) 手部检测框（像素坐标）。

        Returns:
            (768,) float32 特征向量。ViT 未加载时返回零向量。
        """
        if not self._loaded:
            self.load()

        if self._model is None or self._processor is None:
            return np.zeros(self.feature_dim, dtype=np.float32)

        crop = self._crop_hand(frame, bbox)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        inputs = self._processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            features = outputs.pooler_output.squeeze(0)  # (768,)

        return features.cpu().numpy().astype(np.float32)

    def extract_batch(
        self, frame: np.ndarray, hands_data: list[dict]
    ) -> list[np.ndarray]:
        """
        对一帧中的多只手批量提取 ViT 特征。

        Args:
            frame: (H, W, 3) BGR 图像。
            hands_data: detect() 返回的手部列表。

        Returns:
            与 hands_data 等长的特征列表。置信度 <0.3 的手返回零向量。
        """
        features: list[np.ndarray] = []
        zero = np.zeros(self.feature_dim, dtype=np.float32)

        for hand in hands_data:
            if hand.get("confidence", 0) < 0.3:
                features.append(zero)
                continue
            try:
                feat = self.extract(frame, hand["bbox"])
                features.append(feat)
            except Exception:
                logger.debug("ViT 特征提取失败，使用零向量", exc_info=True)
                features.append(zero)

        return features

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _crop_hand(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """裁剪手部区域并 resize 到 CROP_SIZE x CROP_SIZE。"""
        x, y, w, h = bbox
        h_frame, w_frame = frame.shape[:2]

        # 扩展边界（15% margin）
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w_frame, x + w + margin_x)
        y2 = min(h_frame, y + h + margin_y)

        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (self.CROP_SIZE, self.CROP_SIZE))
        return crop

    # ------------------------------------------------------------------
    # 上下文管理器
    # ------------------------------------------------------------------

    def __enter__(self) -> "ViTFeatureExtractor":
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.unload()
