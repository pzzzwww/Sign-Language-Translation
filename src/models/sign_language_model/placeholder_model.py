from __future__ import annotations

from pathlib import Path

from src.interfaces import SignLanguageModel


class PlaceholderSignLanguageModel(SignLanguageModel):
    """
    手语识别占位模型。

    sign_language_model/ 目录中的真实模型尚未接入时使用此实现。
    predict() 会抛出 NotImplementedError，mock 模式由 SignService 处理。

    替换真实模型时：
        1. 在 models/sign_language_model/ 中新建实现文件
        2. 继承 SignLanguageModel，实现四个抽象方法
        3. 在 models/__init__.py 中更新 register_sign_language_model()
        业务代码零改动。
    """

    def load(self) -> None:
        pass

    def predict(self, frames: list[str | Path]) -> list[str]:
        raise NotImplementedError(
            "手语识别模型尚未实现。\n"
            "请在 models/sign_language_model/ 中创建真实模型实现，"
            "并通过 models.register_sign_language_model() 注册后使用。"
        )

    def unload(self) -> None:
        pass

    def is_loaded(self) -> bool:
        return True
