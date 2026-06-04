"""
说话人性别识别服务。

基于 MediaPipe 手部检测参数返回性别判断。
当前实现为默认模式：
  - 提供性别接口供 TTS 语音选择使用
  - 默认使用女声（Microsoft Huihui）

扩展方式：
  子类化 GenderService 并覆盖 detect() 方法即可接入真实性别识别模型。
"""

from __future__ import annotations


class GenderService:
    """
    性别识别服务。

    当前版本固定返回女声（中文语音只有 Huihui 女声可用）。
    子类可重写 detect() 接入真实人脸/语音性别识别。
    """

    def detect(self, tokens: list[str] | None = None) -> str:
        """
        识别说话人性别。

        Args:
            tokens: 可选，手语 Token 列表（用于辅助判断）。

        Returns:
            "male" 或 "female"
        """
        # 当前固定返回 female（Microsoft Huihui 中文女声）
        # 后续可接入人脸检测或语音分析实现自动识别
        return "female"
