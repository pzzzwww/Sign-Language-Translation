"""
【知识点：模型适配器 / 外观模式 (Facade Pattern)】

这个文件是三个子模块的"指挥官"：
  MediaPipe 检测手 → ViT 提取视觉特征(可选) → CSLTransformer 时序分类
它把三个模块串成一条完整的识别管线，对外暴露统一的 SignLanguageModel 接口。

设计亮点：
  - 模块解耦：MediaPipe / ViT / CSL 各自独立，任何一环可单独替换
  - 可选启用 ViT：USE_VIT 控制，加载失败自动回退，不影响其他模块
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
from src.models.sign_language_model.vit_encoder import ViTFeatureExtractor
from src.config import SIGN_MODEL_PATH, USE_VIT, CSL_INPUT_DIM, MEDIAPIPE_MIN_DETECTION, MEDIAPIPE_MIN_TRACKING

logger = logging.getLogger(__name__)


def build_hands_feature(hands_data: list[dict]) -> np.ndarray:
    """【知识点：双手特征拼接 + 坐标系归一化】

    将 MediaPipe 检测结果转换为模型输入向量。
    处理的关键问题：
      1. 左手镜像：左右手在图像中是镜像关系，需要翻转 x 坐标统一坐标系
         MediaPipe 左手 x 坐标 → 镜像 1.0-x → 变为"右手坐标系"
      2. 缺失手填零：检测不到某只手时用零向量占位
      3. 拼接顺序：左手在前 [0:63]，右手在后 [63:126]

    格式: [左手63维(镜像) | 右手63维]
    """
    if CSL_INPUT_DIM == 63:
        # 兼容旧 63 维模式：只取置信度最高的手
        best = None
        best_conf = 0.0
        for h in hands_data:
            if h.get("confidence", 0) > best_conf:
                best = h
                best_conf = h["confidence"]
        if best is not None:
            lm = best["landmarks"].copy()
            if best.get("handedness") == "Left":
                lm[:, 0] = 1.0 - lm[:, 0]
            return landmarks_to_flatten(lm)
        return np.zeros(63, dtype=np.float32)

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

    return np.concatenate([left_vec, right_vec])


class RealSignLanguageModel(SignLanguageModel):
    """
    手语识别模型（MediaPipe + ViT-B/16 + CSL Transformer）。

    特性:
      - MediaPipe 手部检测 + 21 关键点提取
      - ViT-B/16 手部区域视觉特征（可选，config.USE_VIT 控制）
      - CSL Transformer Encoder 时序分类（自注意力替代 BiLSTM）
      - 支持实时增量识别 + 消融实验（启用/禁用 ViT）
      - 输出检测框/关键点数据供前端 Canvas 渲染
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        csl_model_path: str | Path | None = None,
        use_vit: bool | None = None,
    ) -> None:
        self._model_path = Path(model_path) if model_path else SIGN_MODEL_PATH
        self._csl_model_path = (
            Path(csl_model_path) if csl_model_path
            else Path(__file__).parent / "pretrained" / "csl_model.pt"
        )
        self._use_vit = use_vit if use_vit is not None else USE_VIT

        self._detector: MediaPipeHandDetector | None = None
        self._recognizer: CSLRecognizer | None = None
        self._vit_encoder: ViTFeatureExtractor | None = None
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

        # ViT 视觉编码器（可选）
        if self._use_vit:
            logger.info("初始化 ViT-B/16 视觉编码器...")
            try:
                self._vit_encoder = ViTFeatureExtractor(device="cpu")
                self._vit_encoder.load()
                logger.info("  ✅ ViT-B/16 就绪 (特征维度=%d)", self._vit_encoder.feature_dim)
            except Exception:
                logger.warning(
                    "  ⚠️ ViT-B/16 加载失败，回退到仅关键点模式", exc_info=True,
                )
                self._use_vit = False
                self._vit_encoder = None
        else:
            logger.info("ViT 视觉编码器: 未启用（config.USE_VIT=False）")

        logger.info("初始化 CSL 手语识别器...")
        vit_dim = self._vit_encoder.feature_dim if self._vit_encoder else 768
        self._recognizer = CSLRecognizer(
            model_path=self._csl_model_path,
            confidence_threshold=0.4,
            stability_threshold=2,
            cooldown_frames=15,
            use_vit=self._use_vit,
            vit_dim=vit_dim,
        )
        try:
            self._recognizer.load()
            diag = self._recognizer.get_diagnostic_info()
            logger.info("  ✅ CSL 识别器加载完成")
            logger.info("  识别模式: %s", diag["model_type"])
            logger.info("  已训练权重: %s", diag["has_trained_weights"])
            logger.info("  词汇表大小: %d", diag["vocabulary_size"])
            logger.info("  输入维度: %d (关键点63 %s)",
                       63 + (vit_dim if self._use_vit else 0),
                       "+ ViT%d" % vit_dim if self._use_vit else "")
        except Exception:
            logger.warning(
                "  ⚠️ CSL 模型加载异常，使用启发式识别模式", exc_info=True,
            )

        self._loaded = True
        logger.info(
            "手语识别模型就绪（MediaPipe + %s + CSL Transformer）",
            "ViT-B/16" if self._use_vit else "关键点"
        )
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

            # 双手拼接特征 + ViT（取最佳手）
            feature = build_hands_feature(hands_data)
            vit_feat: np.ndarray | None = None
            if self._use_vit and self._vit_encoder is not None and hands_data:
                best = max(hands_data, key=lambda h: h.get("confidence", 0))
                if best["confidence"] >= 0.3:
                    try:
                        vit_feat = self._vit_encoder.extract(image, best["bbox"])
                    except Exception:
                        pass

            token = recognizer.classify_frame(
                feature,
                confidence_hint=hands_data[0]["confidence"] if hands_data else 0.0,
                vit_features=vit_feat,
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

        # ViT 特征（取置信度最高的手部区域）
        vit_feat: np.ndarray | None = None
        if self._use_vit and self._vit_encoder is not None and hands_data:
            best_hand = max(hands_data, key=lambda h: h.get("confidence", 0))
            if best_hand["confidence"] >= 0.3:
                try:
                    vit_feat = self._vit_encoder.extract(image, best_hand["bbox"])
                except Exception:
                    pass

        token = self._recognizer.classify_frame(
            feature,
            confidence_hint=avg_conf,
            vit_features=vit_feat,
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
        if self._vit_encoder is not None:
            self._vit_encoder.unload()
            self._vit_encoder = None
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
    def vit_encoder(self) -> ViTFeatureExtractor | None:
        return self._vit_encoder

    @property
    def using_vit(self) -> bool:
        return self._use_vit and self._vit_encoder is not None
