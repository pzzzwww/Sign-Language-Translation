"""
============================================================
语义句子生成器 — 将手势词序列组合成自然中文句子
============================================================
"""
import re
from typing import List, Tuple


class SentenceComposer:
    """
    将一系列手语词汇组合为语义通顺的中文句子

    策略:
    1. 去重: 连续相同词汇合并
    2. 语法模板: 匹配常见中文句式
    3. 短语组合: 识别词组形成完整表达
    """

    # 常见中文短语模板
    PHRASES = [
        (["我", "爱", "你"], "我爱你"),
        (["你", "好"], "你好"),
        (["谢", "谢"], "谢谢"),
        (["对", "不", "起"], "对不起"),
        (["没", "关", "系"], "没关系"),
        (["早", "上", "好"], "早上好"),
        (["晚", "上", "好"], "晚上好"),
        (["再", "见"], "再见"),
        (["不", "客", "气"], "不客气"),
        (["喜", "欢"], "喜欢"),
        (["高", "兴"], "高兴"),
        (["可", "以"], "可以"),
        (["帮", "助"], "帮助"),
        (["工", "作"], "工作"),
        (["学", "校"], "学校"),
        (["朋", "友"], "朋友"),
        (["知", "道"], "知道"),
        (["不", "知", "道"], "不知道"),
        (["没", "有"], "没有"),
        (["什", "么"], "什么"),
        (["怎", "么", "样"], "怎么样"),
    ]

    # 语义模板: 词序列 → 完整句子
    TEMPLATES = [
        # 问候类
        (["你好"], "{0}！"),
        (["你好", "高兴", "见"], "你好，很高兴见到你！"),
        # 陈述类
        (["我", "*", "你"], "我{1}你"),
        (["我", "是", "*"], "我是{2}"),
        (["我", "要", "*"], "我要{2}"),
        (["我", "想", "*"], "我想{2}"),
        (["我", "喜欢", "*"], "我喜欢{2}"),
        (["我", "不", "*"], "我不{2}"),
        (["你", "是", "*"], "你是{2}？"),
        (["你", "要", "*"], "你要{2}吗？"),
        (["你", "好", "吗"], "你好吗？"),
        (["他", "是", "*"], "他是{2}"),
        # 问答类
        (["什么"], "什么？"),
        (["为什么"], "为什么？"),
        (["怎么", "*"], "怎么{1}？"),
        (["可以", "吗"], "可以吗？"),
        (["可以"], "可以！"),
        (["不", "可以"], "不可以。"),
        (["是", "的"], "是的。"),
        (["不", "是"], "不是。"),
        # 动作类
        (["我", "去", "*"], "我去{2}"),
        (["我", "来", "了"], "我来了！"),
        (["我", "吃", "*"], "我吃{2}"),
        (["我", "喝", "*"], "我喝{2}"),
        # 状态类
        (["我", "高兴"], "我很高兴！"),
        (["我", "难过"], "我很难过。"),
        (["我", "好"], "我很好。"),
        (["我", "不", "好"], "我不好。"),
        # 数字相关
        (["一", "二", "三"], "一、二、三！"),
    ]

    def __init__(self):
        self.raw_words = []
        self.current_sentence = ""

    def add_word(self, word: str):
        """添加一个识别到的词"""
        # 去重: 连续相同词忽略
        if self.raw_words and self.raw_words[-1] == word:
            return
        self.raw_words.append(word)

    def build(self) -> str:
        """将当前词序列组合为自然句子"""
        if not self.raw_words:
            return ""

        # 尝试模板匹配
        result = self._match_template(self.raw_words.copy())

        # 如果模板没匹配上, 直接拼接
        if result is None:
            result = "".join(self.raw_words)

        return self._add_punctuation(result)

    def finalize(self) -> str:
        """结束当前句子并返回"""
        sentence = self.build()
        self.raw_words.clear()
        self.current_sentence = ""
        return sentence

    def get_current(self) -> str:
        """获取当前正在构建的句子(预览)"""
        if not self.raw_words:
            return ""
        return "".join(self.raw_words)

    def _match_template(self, words: List[str]) -> str:
        """尝试用语义模板匹配词序列"""
        # 先尝试完整短语匹配
        for pattern, replacement in self.PHRASES:
            plen = len(pattern)
            for i in range(len(words) - plen + 1):
                if words[i:i+plen] == pattern:
                    # 替换短语
                    words = words[:i] + [replacement] + words[i+plen:]
                    break

        # 再尝试句法模板
        for pattern, template in self.TEMPLATES:
            if self._fuzzy_match(words, pattern):
                return self._fill_template(words, pattern, template)

        return None

    def _fuzzy_match(self, words: List[str], pattern: List[str]) -> bool:
        """模糊匹配: * 匹配任意单字"""
        if len(words) != len(pattern):
            return False
        for w, p in zip(words, pattern):
            if p != "*" and w != p:
                return False
        return True

    def _fill_template(self, words: List[str], pattern: List[str], template: str) -> str:
        """填充模板: {0}→words[0], {1}→words[1], ..."""
        result = template
        for i, w in enumerate(words):
            result = result.replace(f"{{{i}}}", w)
        return result

    def _add_punctuation(self, text: str) -> str:
        """添加合适的标点"""
        if not text:
            return text
        # 已有标点则跳过
        if text[-1] in "。！？，、；：":
            return text
        # 疑问词加问号
        if any(q in text for q in ["吗", "什么", "怎么", "谁", "哪"]):
            return text + "？"
        # 感叹词
        if any(e in text for e in ["好", "棒", "高兴", "喜欢", "爱"]):
            return text + "！"
        return text + "。"

    def clear(self):
        self.raw_words.clear()
        self.current_sentence = ""


class SemanticRewriter:
    """
    后处理语义改写器
    将简单拼接的词序列改写为更自然的表达
    """

    REWRITE_RULES = [
        # (原始模式, 改写后)
        ("我你", "我和你"),
        ("你我", "你和我"),
        ("我去学校学习", "我去学校学习"),
        ("我吃饭", "我吃饭"),
        ("我喝水", "我喝水"),
        ("你好吗我很好", "你好吗？我很好。"),
        ("我爱你谢谢", "我爱你，谢谢！"),
        ("对不起没关系", "对不起，没关系。"),
        ("你好谢谢", "你好，谢谢！"),
        ("可以吗不可以", "可以吗？不可以。"),
    ]

    @classmethod
    def rewrite(cls, text: str) -> str:
        """改写文本使其更自然"""
        for pattern, replacement in cls.REWRITE_RULES:
            if text == pattern:
                return replacement
        return text
