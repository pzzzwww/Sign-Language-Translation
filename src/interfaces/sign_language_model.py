"""
【知识点：接口层 / 抽象基类】

这是整个项目的模型接口层，定义了“手语识别模型”的契约。
任何实现这个接口的模型，都可以无缝接入系统。

涉及知识点：
  - ABC (Abstract Base Class): Python 的抽象基类，强制子类实现特定方法
  - @abstractmethod: 装饰器，标记子类必须实现的方法
  - 策略模式 (Strategy Pattern): 通过统一接口封装不同算法实现，运行时可互换
  - __enter__ / __exit__: Python 上下文管理器协议，支持 with 语句自动管理资源
"""
from abc import ABC, abstractmethod
from pathlib import Path


class SignLanguageModel(ABC):
    """手语识别模型统一接口。

    输入：视频帧图像路径列表
    输出：识别出的手语词汇列表（乱序，交给 TextTranslateModel 重组）

    替换模型时只需实现本接口，业务代码无需改动。
    """

    @abstractmethod
    def load(self) -> None:
        """将模型权重加载到内存/显存，幂等（重复调用无副作用）。"""

    @abstractmethod
    def predict(self, frames: list[str | Path]) -> list[str]:
        """从视频帧列表识别手语词汇。"""

    @abstractmethod
    def unload(self) -> None:
        """释放模型占用的内存/显存资源。"""

    @abstractmethod
    def is_loaded(self) -> bool:
        """返回模型是否已加载到内存。"""

    # ---------- 上下文管理器（with 语句自动 load / unload）----------

    def __enter__(self) -> "SignLanguageModel":
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.unload()
