"""
手语识别模型工厂模块。

业务代码只需调用工厂函数，无需关心具体实现类。
替换模型时调用 register_sign_language_model() 注册新实现即可，业务代码零改动。

用法示例：
    from src.models import get_sign_language_model

    model = get_sign_language_model()
    model.load()
    tokens = model.predict(frame_paths)
"""

from __future__ import annotations

from typing import Type

from src.interfaces import SignLanguageModel
from src.models.sign_language_model import RealSignLanguageModel

_sign_language_cls: Type[SignLanguageModel] = RealSignLanguageModel

# 单例缓存：确保所有代码拿到同一个模型实例
_sign_instance: SignLanguageModel | None = None


def get_sign_language_model(**kwargs) -> SignLanguageModel:
    """返回当前注册的 SignLanguageModel 单例（不自动加载）。"""
    global _sign_instance
    if _sign_instance is None:
        _sign_instance = _sign_language_cls(**kwargs)
    return _sign_instance


def register_sign_language_model(cls: Type[SignLanguageModel]) -> None:
    """注册新的手语识别模型实现类。"""
    if not (isinstance(cls, type) and issubclass(cls, SignLanguageModel)):
        raise TypeError(f"{cls} 必须是 SignLanguageModel 的子类")
    global _sign_language_cls
    _sign_language_cls = cls


__all__ = [
    "get_sign_language_model",
    "register_sign_language_model",
]
