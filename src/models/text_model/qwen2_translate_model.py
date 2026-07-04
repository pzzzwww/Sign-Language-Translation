from __future__ import annotations

import torch

from src.interfaces import TextTranslateModel

_SYSTEM_PROMPT = (
    "你是手语识别结果整理助手。"
    "输入是计算机视觉模型识别出的手势词汇序列，可能存在误识别。"
    "请你根据语义常识判断每个词是否合理："
    "如果某个词导致整句不通顺，用语境最匹配的词替换它；"
    "如果词汇顺序不符合中文习惯，自行调整语序；"
    "根据上下文补充必要的连接词和标点。"
    "只输出最终句子，不要解释修改过程。"
)


class Qwen2TranslateModel(TextTranslateModel):
    """
    基于 Qwen2-1.5B-Instruct 的手语词汇重组模型。

    Args:
        model_path: HuggingFace 模型名称或本地模型目录路径。
        max_new_tokens: 生成句子的最大 token 数。
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2-0.5B-Instruct",
        max_new_tokens: int = 40,
    ) -> None:
        self._model_path = model_path
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

        self._model.eval()

    def translate(self, words: list[str]) -> str:
        if not words:
            raise ValueError("words 不能为空列表")
        if not self.is_loaded():
            self.load()

        input_text = " ".join(words)
        user_msg = f"手语识别结果（注意：识别可能有个别错误）：{input_text}"
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
            f"手语识别结果（注意：识别可能有个别错误）：{input_text}\n"
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
