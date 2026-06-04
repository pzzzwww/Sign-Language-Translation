"""
MockTranslateModel — 手语 Token→自然语言模拟翻译器。

当 Qwen2-1.5B 无法加载时自动替代，确保全链路可运行。
不需要加载任何神经网络模型，零内存占用。

Token 映射规则：
  1. 预定义映射表精确匹配（Token001 → "你好"）
  2. 数字手势自动转换（"1" → "数字一"）
  3. 未知 Token 直接透传
"""

from __future__ import annotations

from typing import Optional

from src.interfaces import TextTranslateModel

# Token → 中文文本映射表
_TOKEN_MAP: dict[str, str] = {
    # 数字手势（手语数字 1-5）
    "1": "数字一",
    "2": "数字二",
    "3": "数字三",
    "4": "数字四",
    "5": "数字五",
    # 手语 Token 编号（模拟真实识别输出）
    "Token001": "你好",
    "Token002": "谢谢",
    "Token003": "很高兴",
    "Token004": "认识你",
    "Token005": "今天天气很好",
    "Token006": "请问",
    "Token007": "超市",
    "Token008": "走",
    "Token009": "怎么",
    "Token010": "我",
    "Token011": "喜欢",
    "Token012": "你",
    "Token013": "是不是",
    "Token014": "在",
    "Token015": "哪里",
    # 常见手语词汇（用于流水线联调）
    "你": "你",
    "我": "我",
    "他": "他",
    "好": "好",
    "是": "是",
    "有": "有",
    "要": "要",
    "想": "想",
    "能": "能",
    "会": "会",
    "可以": "可以",
    "喜欢": "喜欢",
    "爱": "爱",
    "知道": "知道",
    "说": "说",
    "看": "看",
    "听": "听",
    "吃": "吃",
    "喝": "喝",
    "去": "去",
    "来": "来",
    "走": "走",
    "手机": "手机",
    "电脑": "电脑",
    "电视": "电视",
    "问题": "问题",
    "东西": "东西",
    "名字": "名字",
    "地方": "地方",
    "时间": "时间",
    "今天": "今天",
    "明天": "明天",
    "昨天": "昨天",
    "请": "请",
    "帮助": "帮助",
    "谢谢": "谢谢",
    "对不起": "对不起",
    "没关系": "没关系",
    "再见": "再见",
    "请问": "请问",
    "超市": "超市",
    "怎么": "怎么",
    "是不是": "是不是",
    "哪里": "哪里",
    "什么": "什么",
    "为什么": "为什么",
    "多少": "多少",
    "非常": "非常",
    "高兴": "高兴",
    "难过": "难过",
    "累": "累",
    "忙": "忙",
    "漂亮": "漂亮",
    "叫": "叫",
    "告诉": "告诉",
    "帮助": "帮助",
    "工作": "工作",
    "学习": "学习",
    "休息": "休息",
    "睡觉": "睡觉",
}

# 常用语序模板（多 Token → 完整句子）
# 覆盖当前 9 个 CSL 手势 (你好/谢谢/对不起/没关系/为什么/谁/你/我/喜欢) 的常见组合
_SENTENCE_TEMPLATES: dict[str, str] = {
    # ---- 未知手势 ----
    "未知手势": "检测到未知手势，请重新比划",
    # ---- 问候 ----
    "你好": "你好！",
    "谢谢": "谢谢！",
    "对不起": "对不起。",
    "没关系": "没关系。",
    # ---- 人称 + 情感 ----
    "我 喜欢 你": "我喜欢你。",
    "你 喜欢 我": "你喜欢我吗？",
    "谢谢 你": "谢谢你。",
    "我 谢谢 你": "我谢谢你。",
    # ---- 问答 ----
    "为什么": "为什么？",
    "你 是 谁": "你是谁？",
    "谁": "谁？",
    "你 为什么 喜欢 我": "你为什么喜欢我？",
    "我 为什么 喜欢 你": "我为什么喜欢你？",
    # ---- 道歉/回应 ----
    "对不起 没关系": "对不起，没关系。",
    "没关系 对不起": "没关系，对不起。",
    "我 对不起 你": "我对不起你。",
    "你 对不起 我": "你对不起我。",
    # ---- 组合 ----
    "你好 谢谢": "你好，谢谢！",
    "谢谢 没关系": "谢谢，没关系。",
    "你好 我 喜欢 你": "你好，我喜欢你。",
    "你 好": "你好！",
    "再见": "再见！",
    # ---- 保留旧模板兼容 ----
    "你 喜欢 我 是不是": "你是不是喜欢我？",
    "请问 超市 走 怎么": "请问超市怎么走？",
    "你 手机 在 哪里": "你的手机在哪里？",
    "我 想 你": "我想你。",
    "今天 天气 很好": "今天天气很好。",
    "你好 谢谢 再见": "你好，谢谢，再见！",
    "我 高兴": "我很高兴。",
    "我 难过": "我很难过。",
    "请 帮助 我": "请帮助我。",
    "你 叫什么 名字": "你叫什么名字？",
    "我 叫": "我叫...",
}


class MockTranslateModel(TextTranslateModel):
    """
    模拟翻译模型 — 基于 Token 映射表的轻量级翻译器。

    不加载任何深度学习模型，仅通过查表完成 Token→中文映射。
    用于：
      - 流水线联调和端到端测试
      - Qwen2 模型不可用时的自动降级
      - 前端 UI 开发调试
    """

    def __init__(self) -> None:
        self._loaded = False

    # ------------------------------------------------------------------
    # TextTranslateModel 接口
    # ------------------------------------------------------------------

    def load(self) -> None:
        self._loaded = True

    def translate(self, words: list[str]) -> str:
        if not words:
            raise ValueError("词汇列表不能为空")

        # 1) 尝试完整匹配句子模板
        key = " ".join(words)
        if key in _SENTENCE_TEMPLATES:
            return _SENTENCE_TEMPLATES[key]

        # 2) 逐 Token 映射
        translated: list[str] = []
        for w in words:
            if w in _TOKEN_MAP:
                translated.append(_TOKEN_MAP[w])
            elif w.startswith("Token"):
                # 未定义的 Token 编号：友善提示
                translated.append(f"[Token{w[5:]}]")
            else:
                # 未知词汇直接透传
                translated.append(w)

        # 3) 智能拼接
        result = " ".join(translated)
        if len(translated) == 1:
            return result

        # 对多词汇结果做简单语法修饰
        last = translated[-1] if translated else ""
        punctuation_ends = {"吗", "呢", "吧", "？", "！", "。"}

        if last in {"是不是", "怎么", "什么", "哪里", "为什么", "多少", "几"}:
            result += "？"
        elif last in {"吗", "呢"}:
            result += "？"
        elif not any(result.endswith(p) for p in punctuation_ends):
            result += "。"

        return result

    def translate_with_emotion(self, words: list[str], emotion_context: str) -> str:
        """
        带情感上下文的翻译。

        Mock 模式下在翻译结果前添加情感前缀。
        """
        result = self.translate(words)
        if emotion_context:
            prefix_map = {
                "高兴": "（开心地）",
                "悲伤": "（难过地）",
                "激动": "（激动地）",
                "愤怒": "（生气地）",
                "平静": "",
            }
            for key, prefix in prefix_map.items():
                if key in emotion_context:
                    if prefix:
                        result = prefix + result
                    break
        return result

    def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded
