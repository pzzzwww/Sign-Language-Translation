"""
模型工厂模块。

业务代码只需调用工厂函数，无需关心具体实现类。
替换模型时调用 register_*() 注册新实现即可，业务代码零改动。

用法示例：
    from src.models import get_text_translate_model

    model = get_text_translate_model()
    sentence = model.translate(["你", "喜欢", "我", "是不是"])

替换模型示例：
    from src.models import register_text_translate_model
    from src.models.text_model.my_new_model import MyNewModel

    register_text_translate_model(MyNewModel)
"""

from __future__ import annotations

from typing import Type

from src.interfaces import SignLanguageModel, TextTranslateModel
from src.models.sign_language_model import RealSignLanguageModel
from src.models.text_model import Qwen2TranslateModel

_sign_language_cls: Type[SignLanguageModel] = RealSignLanguageModel
_text_translate_cls: Type[TextTranslateModel] = Qwen2TranslateModel

# 单例缓存：确保所有代码拿到同一个模型实例
_sign_instance: SignLanguageModel | None = None
_text_instance: TextTranslateModel | None = None


def get_sign_language_model(**kwargs) -> SignLanguageModel:
    """返回当前注册的 SignLanguageModel 单例（不自动加载）。"""
    global _sign_instance
    if _sign_instance is None:
        _sign_instance = _sign_language_cls(**kwargs)
    return _sign_instance


def get_text_translate_model(**kwargs) -> TextTranslateModel:
    """
    返回当前注册的 TextTranslateModel 单例（不自动加载）。

    示例：
        get_text_translate_model()
    """
    global _text_instance
    if _text_instance is None:
        _text_instance = _text_translate_cls(**kwargs)
    return _text_instance


def register_sign_language_model(cls: Type[SignLanguageModel]) -> None:
    """注册新的手语识别模型实现类。"""
    if not (isinstance(cls, type) and issubclass(cls, SignLanguageModel)):
        raise TypeError(f"{cls} 必须是 SignLanguageModel 的子类")
    global _sign_language_cls
    _sign_language_cls = cls


def register_text_translate_model(cls: Type[TextTranslateModel]) -> None:
    """注册新的文本翻译模型实现类。"""
    if not (isinstance(cls, type) and issubclass(cls, TextTranslateModel)):
        raise TypeError(f"{cls} 必须是 TextTranslateModel 的子类")
    global _text_translate_cls
    _text_translate_cls = cls


__all__ = [
    "get_sign_language_model",
    "get_text_translate_model",
    "register_sign_language_model",
    "register_text_translate_model",
]
