"""
手势识别器 — 基于 Hand-Gesture-19 (SigLIP2) 预训练模型
19个手势类, 98.3%准确率, 93M参数
"""
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


class GestureRecognizer:
    """Hand-Gesture-19 手势识别器"""

    DEFAULT_MODEL_PATH = Path(__file__).parent / "pretrained" / "hand_gesture_19"

    def __init__(self, model_path: str | Path | None = None):
        self._model_path = Path(model_path) if model_path else self.DEFAULT_MODEL_PATH
        self._model = None
        self._processor = None
        self._loaded = False

        # 稳定化
        self._last_label = None
        self._stability_count = 0
        self.stability_threshold = 4

        # 句子构建
        self._sentence_words = []
        self._last_word = None
        self._word_cooldown = 0
        self._word_cd = 20
        self._finish_cooldown = 0
        self._finish_frames = 40

        # 推理节流
        self._tick_count = 0
        self.infer_every = 6

    # 手势标签映射
    LABEL_MAP = {
        0: "打电话", 1: "不喜欢", 2: "握拳",
        3: "四", 4: "喜欢", 5: "静音",
        6: "无手势", 7: "OK",
        8: "一", 9: "手掌",
        10: "胜利", 11: "胜利(反)",
        12: "摇滚", 13: "停止",
        14: "停止(反)", 15: "三",
        16: "三(2)", 17: "二",
        18: "二(反)",
    }

    def load(self):
        if self._loaded:
            return

        from transformers import AutoImageProcessor, SiglipForImageClassification

        logger.info("加载 Hand-Gesture-19 模型...")

        # 检查模型权重是否已下载
        model_file = self._model_path / "model.safetensors"
        if not (model_file.exists() and model_file.stat().st_size > 100):
            logger.warning(
                "Hand-Gesture-19 权重未下载。正尝试自动下载 (~355MB)..."
            )
            self._auto_download()

        try:
            self._model = SiglipForImageClassification.from_pretrained(
                str(self._model_path), local_files_only=True)
            self._processor = AutoImageProcessor.from_pretrained(
                str(self._model_path), local_files_only=True)
        except (OSError, FileNotFoundError):
            logger.warning(
                "模型加载失败，可能是权重文件损坏。尝试重新下载...\n"
                "请手动运行: python models/sign_language_model/download_hand_gesture.py"
            )
            raise

        self._loaded = True
        logger.info("Hand-Gesture-19 加载完成 (19手势, 93M参数)")

    def _auto_download(self) -> None:
        """触发自动下载模型权重。"""
        from models.sign_language_model.download_hand_gesture import main as download_main
        download_main()

    def classify(self, image: np.ndarray) -> Tuple[int, str, float, list]:
        """
        Args:
            image: (H, W, 3) BGR uint8 ROI图像

        Returns:
            (class_id, label, confidence, top3_list)
        """
        if not self._loaded:
            self.load()

        import torch
        from PIL import Image

        if image.dtype == np.float32:
            image = (image * 255).clip(0, 255).astype(np.uint8)

        # BGR → RGB
        pil = Image.fromarray(image[..., ::-1]).resize((224, 224))
        inputs = self._processor(images=pil, return_tensors="pt")

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).numpy()

        top_indices = np.argsort(-probs)[:3]
        best_id = int(top_indices[0])
        best_label = self.LABEL_MAP.get(best_id, f"class_{best_id}")
        best_conf = float(probs[best_id])

        top3 = []
        for idx in top_indices:
            idx = int(idx)
            top3.append((self.LABEL_MAP.get(idx, f"class_{idx}"), float(probs[idx])))

        return best_id, best_label, best_conf, top3

    # === 句子构建 ===
    def get_stable_word(self, label_id, label, confidence):
        if confidence < 0.4:
            self._stability_count = max(0, self._stability_count - 1)
            return None
        if label_id == self._last_label:
            self._stability_count += 1
        else:
            self._last_label = label_id
            self._stability_count = 1
        if self._stability_count >= self.stability_threshold:
            return label
        return None

    def tick(self):
        self._tick_count += 1
        if self._word_cooldown > 0: self._word_cooldown -= 1
        if self._finish_cooldown > 0: self._finish_cooldown -= 1

    def should_infer(self):
        if self._tick_count >= self.infer_every:
            self._tick_count = 0
            return True
        return False

    def add_word(self, word):
        if word == self._last_word: return None
        if self._word_cooldown > 0: return None
        self._sentence_words.append(word)
        self._last_word = word
        self._word_cooldown = self._word_cd
        self._finish_cooldown = 0
        return None

    def trigger_finish(self):
        if self._sentence_words:
            self._finish_cooldown = self._finish_frames

    def check_finish(self):
        if self._finish_cooldown > 0:
            self._finish_cooldown -= 1
            if self._finish_cooldown <= 0 and self._sentence_words:
                sentence = "".join(self._sentence_words)
                self._sentence_words.clear()
                self._last_word = None
                return sentence
        return None

    def get_current_sentence(self):
        return "".join(self._sentence_words) if self._sentence_words else ""

    def clear_all(self):
        self._sentence_words.clear()
        self._last_word = None
        self._word_cooldown = 0
        self._finish_cooldown = 0
        self._last_label = None
        self._stability_count = 0

    @property
    def model_loaded(self): return self._loaded
    @property
    def num_gestures(self): return 19
