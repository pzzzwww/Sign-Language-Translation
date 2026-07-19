from __future__ import annotations

import torch

from src.interfaces import TextTranslateModel

_SYSTEM_PROMPT = (
    "你是手语词汇连词成句助手。"
    "输入是手语识别出的中文词汇序列，你需要排成通顺的中文句子。"
    "严格遵守："
    "1. 必须使用输入的全部词汇，一个都不能少；"
    "2. 不得把任何词替换成同义词（如'喜欢'不能改'爱'，'对不起'不能改'抱歉'）；"
    "3. 不得新增输入中没有的实义词（名词、动词、形容词等）；"
    "4. 只允许调整词序、补标点、补少量功能词（的、了、吗、呢、是）；"
    "5. 只输出最终句子，不要解释。"
)

# Few-shot 示例：对 0.5B 小模型，示例比规则更有效
_FEW_SHOT: list[dict] = [
    {"role": "user", "content": "手语识别结果：你好"},
    {"role": "assistant", "content": "你好！"},
    {"role": "user", "content": "手语识别结果：我 喜欢 你"},
    {"role": "assistant", "content": "我喜欢你。"},
    {"role": "user", "content": "手语识别结果：你 喜欢 我"},
    {"role": "assistant", "content": "你喜欢我吗？"},
    {"role": "user", "content": "手语识别结果：你 是 谁"},
    {"role": "assistant", "content": "你是谁？"},
    {"role": "user", "content": "手语识别结果：请 帮助 我"},
    {"role": "assistant", "content": "请帮助我。"},
]


class Qwen2TranslateModel(TextTranslateModel):
    """
    基于 Qwen2-0.5B-Instruct 的手语词汇重组模型。

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
            self._model_path, trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        self._model.eval()

    def translate(self, words: list[str]) -> str:
        if not words:
            raise ValueError("words 不能为空列表")
        if not self.is_loaded():
            self.load()

        input_text = " ".join(words)
        user_msg = f"手语识别结果：{input_text}"
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *_FEW_SHOT,
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
                pad_token_id=self._tokenizer.eos_token_id,
            )

        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
