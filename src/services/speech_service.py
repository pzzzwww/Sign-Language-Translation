"""
语音合成服务。

封装 pyttsx3，将文本合成为 WAV 格式音频。
支持根据性别选择语音。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pyttsx3

from src.config import TTS_RATE


class SpeechService:
    """
    语音合成服务。

    封装 pyttsx3，支持性别选择。
    """

    def __init__(self, rate: int = TTS_RATE) -> None:
        self._rate = rate

    def synthesize_to_file(self, text: str, output_path: str | Path, gender: str = "female") -> float:
        """
        将文本合成 WAV 并永久保存到指定路径。

        Args:
            text: 待合成的中文文本。
            output_path: 输出文件路径。
            gender: "male" 或 "female"（用于选择语音）。

        Returns:
            估算的音频时长（秒）。
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine_save(text, str(output_path), gender=gender)
        duration = self._estimate_duration(text)
        return duration

    def synthesize(self, text: str, gender: str = "female") -> bytes:
        """
        将文本合成为 WAV 音频字节（临时文件，自动删除）。

        Returns:
            WAV 格式的音频 bytes。
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        try:
            self._engine_save(text, tmp_path, gender=gender)
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            if not audio_bytes:
                raise RuntimeError("语音合成结果为空")
            return audio_bytes
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _engine_save(self, text: str, output_path: str, gender: str = "female") -> None:
        """初始化引擎并保存到文件，根据性别选择语音。"""
        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        voices = engine.getProperty("voices")

        # 性别 → 语音索引映射
        # voices[0] = Microsoft Huihui (中文女声)
        # voices[1] = Microsoft Zira (英文女声)
        if gender == "male" and len(voices) > 2:
            vid = 2  # 如果有男声则用
        else:
            vid = 0  # 默认女声 Huihui

        vid = min(vid, len(voices) - 1) if voices else 0
        engine.setProperty("voice", voices[vid].id)
        engine.save_to_file(text, output_path)
        engine.runAndWait()

    def _estimate_duration(self, text: str) -> float:
        """根据字数和语速估算音频时长（秒）。"""
        char_count = len(text)
        chars_per_second = self._rate / 60.0
        estimated = char_count / chars_per_second
        return max(estimated, 0.5)  # 最少 0.5 秒
