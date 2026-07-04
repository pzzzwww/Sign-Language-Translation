"""
【知识点：模型适配器 / 外观模式 (Facade Pattern)】

将 MediaPipe 手部检测和 CSLTransformer 时序分类串成完整识别管线，
对外暴露统一的 SignLanguageModel 接口。

设计亮点：
  - 模块解耦：MediaPipe / CSL 各自独立，任何一环可单独替换
  - 双手支持：左右手关键点拼接为 126 维统一输入

实现 SignLanguageModel 统一接口。
同时输出手部检测数据（检测框、关键点）供前端 Canvas 渲染。
"""

from __future__ import annotations
import logging
from pathlib import Path
import cv2
import numpy as np
from src.interfaces import SignLanguageModel
from src.models.sign_language_model.mediapipe_detector import (
    MediaPipeHandDetector,
    landmarks_to_flatten,
)
from src.models.sign_language_model.csl_recognizer import CSLRecognizer
from src.config import MEDIAPIPE_MIN_DETECTION, MEDIAPIPE_MIN_TRACKING

logger = logging.getLogger(__name__)


def build_hands_feature(hands_data: list[dict]) -> np.ndarray:
    """
    把 MediaPipe 检测到的手变成模型能吃的 126 维向量。
    三件事：
      ① 左手 x 坐标镜像（1.0 - x）→ 左手变"右手"，统一坐标系
      ② 每只手 21 关键点 × (x,y,z) = 63 维，左手放前 63，右手放后 63
      ③ 没检测到的手填 63 个零
    """
    left_vec = np.zeros(63, dtype=np.float32)
    right_vec = np.zeros(63, dtype=np.float32)

    for h in hands_data:
        if h.get("confidence", 0) < 0.3:
            continue
        lm = h["landmarks"].copy()
        if h.get("handedness") == "Left":
            lm[:, 0] = 1.0 - lm[:, 0]
            left_vec = landmarks_to_flatten(lm)
        else:
            right_vec = landmarks_to_flatten(lm)

    return np.concatenate([left_vec, right_vec])#返回126维向量


class RealSignLanguageModel(SignLanguageModel):
    """
    手语识别模型（MediaPipe + CSL Transformer）。

    特性:
      - MediaPipe 手部检测 + 21 关键点提取
      - CSL Transformer Encoder 时序分类
      - 双手支持：左右手关键点拼接为 126 维统一输入
      - 输出检测框/关键点数据供前端 Canvas 渲染
    """

    def __init__(
        self,
        csl_model_path: str | Path | None = None,
    ) -> None:
        self._csl_model_path = (
            Path(csl_model_path) if csl_model_path
            else Path(__file__).parent / "pretrained" / "csl_model.pt"
        )

        self._detector: MediaPipeHandDetector | None = None
        self._recognizer: CSLRecognizer | None = None
        self._loaded = False

    # ------------------------------------------------------------------
    # 接口实现
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self._loaded:
            return

        logger.info("=" * 50)
        logger.info("初始化 MediaPipe 手部检测器...")
        try:
            self._detector = MediaPipeHandDetector(
                min_detection_confidence=MEDIAPIPE_MIN_DETECTION,
                min_tracking_confidence=MEDIAPIPE_MIN_TRACKING,
                max_num_hands=2,
            )
            logger.info("  ✅ MediaPipe 手部检测器初始化成功")
        except ImportError as e:
            logger.error("  ❌ MediaPipe 初始化失败: %s", e)
            raise

        logger.info("初始化 CSL 手语识别器...")
        self._recognizer = CSLRecognizer(
            model_path=self._csl_model_path,
            confidence_threshold=0.4,
            stability_threshold=2,
            cooldown_frames=15,
        )
        try:
            self._recognizer.load()
            diag = self._recognizer.get_diagnostic_info()
            logger.info("  ✅ CSL 识别器加载完成")
            logger.info("  识别模式: %s", diag["model_type"])
            logger.info("  已训练权重: %s", diag["has_trained_weights"])
            logger.info("  词汇表大小: %d", diag["vocabulary_size"])
        except Exception:
            logger.warning(
                "  ⚠️ CSL 模型加载异常，使用启发式识别模式", exc_info=True,
            )

        self._loaded = True
        logger.info("手语识别模型就绪（MediaPipe + CSL Transformer）")
        logger.info("=" * 50)

    def predict(self, frames: list[str | Path]) -> list[str]:
        """对帧序列进行手语识别，返回 Token 列表。"""
        if not self._loaded or self._recognizer is None or self._detector is None:
            self.load()

        if self._recognizer is None or self._detector is None:
            return []

        recognizer = self._recognizer
        recognizer.clear()

        tokens: list[str] = []

        for frame_path in frames:
            image = cv2.imread(str(frame_path))
            if image is None:
                continue

            hands_data = self._detector.detect(image)
            if not hands_data:
                continue

            # 双手拼接特征
            feature = build_hands_feature(hands_data)

            token = recognizer.classify_frame(
                feature,
                confidence_hint=hands_data[0]["confidence"] if hands_data else 0.0,
            )
            if token:
                tokens.append(token)

        all_tokens = recognizer.get_tokens()
        return list(dict.fromkeys(all_tokens))

    def predict_frame(self, image: np.ndarray) -> dict:
        """对单帧图像进行检测 + 识别（实时摄像头模式）。"""
        if not self._loaded or self._recognizer is None or self._detector is None:
            self.load()

        if self._recognizer is None or self._detector is None:
            return {"hands_data": [], "tokens": [], "all_tokens": []}

        hands_data = self._detector.detect(image)
        new_tokens: list[str] = []

        frame_hint = getattr(self._recognizer, '_frame_count', 0)
        if frame_hint % 30 == 0:
            if hands_data:
                logger.info(
                    "🖐 MediaPipe检测: %d只手 | conf=%.2f | 缓冲帧=%d | "
                    "累计Token=%d | 模式=%s",
                    len(hands_data),
                    hands_data[0]["confidence"] if hands_data else 0,
                    len(self._recognizer._sequence_buffer),
                    len(self._recognizer._tokens),
                    self._recognizer.model_type,
                )
            else:
                logger.info(
                    "❌ MediaPipe检测: 0只手 | 缓冲帧=%d | 累计Token=%d",
                    len(self._recognizer._sequence_buffer),
                    len(self._recognizer._tokens),
                )

        # 没有检测到手 → 不分类，直接返回
        if not hands_data:
            return {
                "hands_data": [],
                "tokens": [],
                "all_tokens": self._recognizer.get_tokens(),
            }

        # 双手拼接为 126 维，单手时缺失手填零
        feature = build_hands_feature(hands_data)
        avg_conf = np.mean([h["confidence"] for h in hands_data])

        token = self._recognizer.classify_frame(
            feature,
            confidence_hint=avg_conf,
        )

        # 将识别到的 token 绑定到每只检测到的手（前端显示用）
        for hand in hands_data:
            hand["token"] = token

        if token:
            new_tokens.append(token)

        return {
            "hands_data": hands_data,
            "tokens": new_tokens,
            "all_tokens": self._recognizer.get_tokens(),
        }

    def get_all_tokens(self) -> list[str]:
        if self._recognizer is None:
            return []
        return self._recognizer.get_tokens()

    def reset_session(self) -> None:
        if self._detector is not None:
            self._detector.reset()
        if self._recognizer is not None:
            self._recognizer.clear()

    def unload(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None
        if self._recognizer is not None:
            self._recognizer.unload()
            self._recognizer = None
        self._loaded = False
        logger.info("手语识别模型已卸载")

    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def detector(self) -> MediaPipeHandDetector | None:
        return self._detector

    @property
    def recognizer(self) -> CSLRecognizer | None:
        return self._recognizer

    @property
    def using_vit(self) -> bool:
        return False
