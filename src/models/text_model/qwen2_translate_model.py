from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

try:
    import torch
    _TORCH_TEXT_AVAILABLE = True
except (ImportError, OSError):
    _TORCH_TEXT_AVAILABLE = False
    torch = None

from src.interfaces import TextTranslateModel

_SYSTEM_PROMPT = (
    "你是手语文本整理助手。将手语识别 token 重组为语义通顺的自然中文句子。"
    "规则：必须使用全部输入 token、不增删替换任何实义词、"
    "根据语境自动添加标点和必要的连接词（如的、了、是、吗、呢）、"
    "只输出最终句子。"
)


class Qwen2LoRAModel(TextTranslateModel):
    """
    基于 Qwen2-1.5B-Instruct + LoRA 微调的手语词汇重组模型。

    Args:
        model_path: HuggingFace 模型名称或本地模型目录路径。
        lora_path:  可选，LoRA 适配器目录路径。为 None 时使用原始基座模型。
        max_new_tokens: 生成句子的最大 token 数。
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2-0.5B-Instruct",
        lora_path: Optional[str | Path] = None,
        max_new_tokens: int = 40,
    ) -> None:
        self._model_path = model_path
        self._lora_path = Path(lora_path) if lora_path else None
        self._max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        if self.is_loaded():
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        use_gpu = torch.cuda.is_available()
        dtype = torch.float16 if use_gpu else torch.float32
        device_map = "auto" if use_gpu else None

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path, trust_remote_code=True        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=True,
        )

        if self._lora_path is not None:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, str(self._lora_path))
            self._model = self._model.merge_and_unload()

        self._model.eval()

    def translate(self, words: list[str]) -> str:
        if not words:
            raise ValueError("words 不能为空列表")
        if not self.is_loaded():
            self.load()

        input_text = " ".join(words)
        user_msg = f"以下是手语识别的乱序词汇，请重组为通顺句子：{input_text}"
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        return self._generate(messages)

    def translate_with_emotion(self, words: list[str], emotion_context: str) -> str:
        """
        带情感上下文的手语词汇重组。

        Args:
            words: 手语识别出的词汇列表。
            emotion_context: 情感上下文提示词（如 "说话者此刻心情愉悦、开心"）。
        """
        if not words:
            raise ValueError("words 不能为空列表")
        if not self.is_loaded():
            self.load()

        input_text = " ".join(words)
        user_msg = (
            f"以下是手语识别的乱序词汇，请重组为通顺句子：{input_text}\n"
            f"{emotion_context}"
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        return self._generate(messages)

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _generate(self, messages: list[dict]) -> str:
        """执行推理并返回结果。"""
        chat_prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(chat_prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
