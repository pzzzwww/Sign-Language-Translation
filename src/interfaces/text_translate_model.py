"""
【知识点：文本翻译接口】

将手语识别的离散词汇重组成通顺的自然语言句子。
这是 NLP 中的"句子重组 / 文本生成"任务。

涉及知识点：
  - 序列到序列 (Seq2Seq): 输入 token 序列 → 输出自然语言文本
  - LoRA 微调: 低秩适配，只训练少量参数就能让大模型学会新任务
  - 模板匹配 vs 模型推理: mock 是规则匹配（快、零依赖），qwen 是神经网络生成（慢、效果好）
"""
from abc import ABC, abstractmethod


class TextTranslateModel(ABC):
    """手语词汇 → 自然语句转换模型统一接口。

    输入：手语识别得到的乱序词汇列表
    输出：语义通顺的中文句子
    """

    @abstractmethod
    def load(self) -> None:
        """将模型权重加载到内存/显存，幂等（重复调用无副作用）。"""

    @abstractmethod
    def translate(self, words: list[str]) -> str:
        """将乱序手语词汇重组为自然中文句子。"""

    def translate_with_emotion(self, words: list[str], emotion_context: str) -> str:
        """带情感上下文的翻译（默认回退到普通 translate）。

        子类可重写以注入情感上下文改进翻译质量。"""
        return self.translate(words)

    @abstractmethod
    def unload(self) -> None:
        """释放模型占用的内存/显存资源。"""

    @abstractmethod
    def is_loaded(self) -> bool:
        """返回模型是否已加载到内存。"""

    # ---------- 上下文管理器（with 语句自动 load / unload）----------

    def __enter__(self) -> "TextTranslateModel":
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.unload()
