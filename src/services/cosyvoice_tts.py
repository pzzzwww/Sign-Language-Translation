"""
CosyVoice2 TTS 服务 — 基于 Transformer + Flow Matching 的语音合成。

CosyVoice2 (阿里通义实验室) 是当前中文 TTS 的 SOTA 方案:
  - 架构: Transformer Encoder + Flow Matching Decoder + HiFi-GAN Vocoder
  - 支持零样本语音克隆、流式合成、情感控制
  - 相比 pyttsx3 规则引擎，音质和自然度有质的提升

Fallback: CosyVoice2 不可用时自动降级为 pyttsx3。

Usage:
    from services.cosyvoice_tts import CosyVoice2TTSService
    tts = CosyVoice2TTSService()
    tts.load()                              # 首次下载 ~1.5GB
    audio_bytes = tts.synthesize("你好世界")
    tts.synthesize_to_file("你好", "output.wav")
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_COSYVOICE_AVAILABLE = False
try:
    import cosyvoice  # noqa: F401
    _COSYVOICE_AVAILABLE = True
except ImportError:
    pass

try:
    import pyttsx3
    _PYTTSX3_AVAILABLE = True
except ImportError:
    _PYTTSX3_AVAILABLE = False


class CosyVoice2TTSService:
    """
    CosyVoice2 语音合成服务。

    优先使用 CosyVoice2 (Transformer + Flow Matching)。
    CosyVoice2 不可用时自动降级为 pyttsx3。

    CosyVoice2 安装:
        pip install cosyvoice
        或从源码:
        git clone https://github.com/FunAudioLLM/CosyVoice.git
        cd CosyVoice && pip install -e .
    """

    COSYVOICE_MODEL = "iic/CosyVoice2-0.5B"  # ModelScope 模型 ID

    def __init__(self, rate: int = 200, use_cosyvoice: bool = True) -> None:
        self._rate = rate
        self._use_cosyvoice = use_cosyvoice and _COSYVOICE_AVAILABLE
        self._model = None
        self._loaded = False

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self._loaded:
            return

        if self._use_cosyvoice:
            try:
                self._load_cosyvoice()
            except Exception:
                logger.warning(
                    "CosyVoice2 加载失败，降级为 pyttsx3", exc_info=True
                )
                self._use_cosyvoice = False

        self._loaded = True

    def _load_cosyvoice(self) -> None:
        """
        加载 CosyVoice2 模型。支持两种安装方式:
          1. pip install cosyvoice (PyPI, 仅 CLI 服务)
          2. git clone https://github.com/FunAudioLLM/CosyVoice && pip install -e .
             或直接从 ModelScope 加载
        """
        # 尝试使用 ModelScope 直接加载模型
        try:
            from modelscope import snapshot_download

            logger.info("从 ModelScope 下载 CosyVoice2: %s", self.COSYVOICE_MODEL)
            model_dir = snapshot_download(self.COSYVOICE_MODEL)
            logger.info("CosyVoice2 模型路径: %s", model_dir)

            # 尝试导入 CosyVoice 推理类
            from cosyvoice.cli.cosyvoice import CosyVoice  # type: ignore[import-untyped]
            self._model = CosyVoice(model_dir, load_jit=False, fp16=False)
            logger.info("CosyVoice2 就绪 (Transformer + Flow Matching)")
        except ImportError:
            raise ImportError(
                "CosyVoice2 模型代码未安装。请从源码安装:\n"
                "  git clone https://github.com/FunAudioLLM/CosyVoice.git\n"
                "  cd CosyVoice && pip install -e .\n"
                "或使用 pip install cosyvoice 后从 ModelScope 加载"
            )

    def unload(self) -> None:
        self._model = None
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def engine_type(self) -> str:
        if not self._loaded:
            return "未加载"
        return "CosyVoice2 (Transformer + Flow Matching)" if self._use_cosyvoice else "pyttsx3 (SAPI5 降级)"

    # ------------------------------------------------------------------
    # 合成接口
    # ------------------------------------------------------------------

    def synthesize(self, text: str, voice_id: str | None = None) -> bytes:
        """
        文本 → WAV 音频字节。

        Args:
            text: 待合成的中文文本。
            voice_id: CosyVoice2 音色 ID（默认女声）。

        Returns:
            WAV 格式的音频 bytes。
        """
        if not text.strip():
            raise ValueError("text 不能为空")

        if not self._loaded:
            self.load()

        if self._use_cosyvoice and self._model is not None:
            return self._synthesize_cosyvoice(text, voice_id)

        if _PYTTSX3_AVAILABLE:
            return self._synthesize_pyttsx3(text)

        raise RuntimeError("无可用的 TTS 引擎")

    def synthesize_to_file(
        self, text: str, output_path: str | Path, voice_id: str | None = None
    ) -> float:
        """
        文本 → WAV 文件。

        Returns:
            音频时长（秒）。
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        audio_bytes = self.synthesize(text, voice_id=voice_id)
        output_path.write_bytes(audio_bytes)

        return self._estimate_duration(text)

    # ------------------------------------------------------------------
    # 内部: CosyVoice2 合成
    # ------------------------------------------------------------------

    def _synthesize_cosyvoice(
        self, text: str, voice_id: str | None = None
    ) -> bytes:
        """使用 CosyVoice2 合成 WAV。"""
        if self._model is None:
            raise RuntimeError("CosyVoice2 模型未加载")

        # CosyVoice2 API: model.inference(text, stream=False)
        # 返回生成器，每次 yield 一个音频片段
        chunks = []
        for chunk in self._model.inference(
            text, stream=False, speed=1.0
        ):
            chunks.append(chunk)

        if not chunks:
            raise RuntimeError("CosyVoice2 合成结果为空")

        # 拼接所有音频片段为 WAV
        import numpy as np
        audio = np.concatenate([c["tts_speech"] for c in chunks], axis=-1)
        return self._numpy_to_wav(audio)

    # ------------------------------------------------------------------
    # 内部: pyttsx3 降级
    # ------------------------------------------------------------------

    def _synthesize_pyttsx3(self, text: str) -> bytes:
        """使用 pyttsx3 合成（降级方案）。"""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            voices = engine.getProperty("voices")
            if voices:
                engine.setProperty("voice", voices[0].id)
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()

            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            if not audio_bytes:
                raise RuntimeError("pyttsx3 合成结果为空")
            return audio_bytes
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _numpy_to_wav(self, audio: "np.ndarray", sample_rate: int = 24000) -> bytes:
        """将 numpy 音频数组编码为 WAV 字节。"""
        import io
        import wave
        import numpy as np

        audio_int16 = (audio * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return buf.getvalue()

    def _estimate_duration(self, text: str) -> float:
        """根据字数估算音频时长（秒）。"""
        # 中文 TTS 约 3-4 字/秒
        chars_per_second = 3.5
        return max(len(text) / chars_per_second, 0.5)

    # ------------------------------------------------------------------
    # 上下文管理器
    # ------------------------------------------------------------------

    def __enter__(self) -> "CosyVoice2TTSService":
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.unload()
