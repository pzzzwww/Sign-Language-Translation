"""
手语词汇→自然语句翻译服务。

支持三种模式（由 config.TRANSLATION_MODE 控制）：
  - "qwen":  使用 Qwen2-0.5B（加载失败则抛异常）
  - "mock":  使用 MockTranslateModel（零依赖，无需模型，始终安全）
  - "auto":  默认。安全优先：直接使用 MockTranslateModel。
             如需启用 Qwen2，请设置 TRANSLATION_MODE=qwen。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.interfaces import TextTranslateModel
from src.models.text_model import MockTranslateModel
from src.config import TRANSLATION_MODE

logger = logging.getLogger(__name__)


class TranslateService:
    """
    手语词汇→自然语句翻译服务。

    懒加载模型，首次 translate() 调用时自动初始化。
    默认使用 MockTranslateModel（零内存占用，不崩溃）。
    如需加载 Qwen2-0.5B，设置 TRANSLATION_MODE=qwen。
    """

    _instance: Optional[TextTranslateModel] = None
    _mode: str = ""

    # 是否使用 LoRA 微调权重（仅 qwen 模式有效）
    use_lora: bool = True

    # ------------------------------------------------------------------
    # 模型管理
    # ------------------------------------------------------------------

    @classmethod
    def _get_model(cls) -> TextTranslateModel:
        if cls._instance is not None:
            return cls._instance

        mode = TRANSLATION_MODE.lower()

        if mode == "qwen":
            cls._instance = cls._load_qwen_or_raise()
            cls._mode = "qwen"
            logger.info("翻译模式: Qwen2-0.5B")
            return cls._instance

        # mock / auto 都走 Mock（安全优先）
        cls._instance = MockTranslateModel()
        cls._instance.load()
        cls._mode = "mock"
        logger.info("翻译模式: Mock（%s）" % ("配置指定" if mode == "mock" else "auto 默认安全模式"))
        return cls._instance

    @classmethod
    def _load_qwen_or_raise(cls) -> TextTranslateModel:
        """加载 Qwen2-0.5B。失败时向上抛异常（不降级）。"""
        from src.config import LORA_PATH
        from src.models.text_model import Qwen2LoRAModel

        from src.config import TEXT_MODEL_NAME
        lora_path = str(LORA_PATH) if cls.use_lora and LORA_PATH.exists() else None
        model = Qwen2LoRAModel(model_path=TEXT_MODEL_NAME, lora_path=lora_path)
        model.load()  # 可能 SegFault — 用户选择 qwen 模式即接受此风险
        return model

    # ------------------------------------------------------------------
    # 翻译接口
    # ------------------------------------------------------------------

    def translate(self, words: list[str]) -> str:
        if not words:
            raise ValueError("翻译词汇列表不能为空")
        model = self._get_model()
        return model.translate(words)

    def unload(self) -> None:
        """释放模型资源。"""
        if TranslateService._instance is not None:
            TranslateService._instance.unload()
            TranslateService._instance = None
            TranslateService._mode = ""

    @property
    def current_mode(self) -> str:
        """返回当前实际使用的翻译模式。"""
        return self._mode if self._mode else "未初始化"
