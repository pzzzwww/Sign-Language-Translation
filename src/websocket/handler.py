"""
WebSocket 连接管理器。

每个客户端使用自己的摄像头 → 发帧到服务端 → 服务端处理 → 返回结果。
每个 WebSocket 连接维护独立的识别会话。
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from src.services.translate_service import TranslateService
from src.services.speech_service import SpeechService
from src.services.cosyvoice_tts import CosyVoice2TTSService
from src.services.gender_service import GenderService
from src.services import history_service
from src.models.sign_language_model.csl_recognizer import CSLRecognizer
from src.models.sign_language_model.real_recognizer import build_hands_feature
from src.config import AUDIO_DIR, CSL_MODEL_PATH


class StreamHandler:
    """每个 WebSocket 连接一个实例，独立识别会话。"""

    def __init__(
        self,
        sign,         # SignService (共享 MediaPipe detector)
        translate: TranslateService,
        speech: SpeechService,
    ) -> None:
        self._sign = sign
        self.translate = translate
        self.speech = speech
        self.gender = GenderService()

        self._ws: WebSocket | None = None
        self._running = False

        # 当前会话状态
        self._current_tokens: list[str] = []
        self._current_sentence: str = ""
        self._current_history_id: int | None = None
        self._current_gender: str = "female"

        # 跳帧优化 + 自动翻译
        self._frame_idx = 0
        self._detect_interval = 1       # 每帧跑 MediaPipe 检测
        self._classify_interval = 1     # 每帧跑 CSL 分类
        self._last_hands: list[dict] = []
        self._last_landmarks: np.ndarray | None = None  # 上一帧特征，用于运动检测
        self._last_token_count = 0       # 跟踪 recognizer 已确认 token 数量
        self._last_tokens_snapshot = ""  # 自动翻译去重
        self._translate_task: asyncio.Task | None = None

        # 每个连接独立的 CSL 识别器（加载训练权重，追踪 Token 序列）
        self._recognizer = CSLRecognizer(
            model_path=CSL_MODEL_PATH,
            confidence_threshold=0.35,
            stability_threshold=2,
            cooldown_frames=8,
            use_vit=False,
        )

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------

    async def handle(self, ws: WebSocket) -> None:
        _log = logging.getLogger(__name__)
        try:
            await ws.accept()
        except WebSocketDisconnect:
            return
        self._ws = ws
        self._running = True

        try:
            loop = asyncio.get_event_loop()
            model = self._sign._model
            if not model.is_loaded():
                await self._send(ws, type="status", state="loading_model",
                                 message="正在加载手语识别模型...")
                await loop.run_in_executor(None, model.load)
            await self._send(ws, type="status", state="loading_model",
                             message="正在加载 CSL 识别器...")
            await loop.run_in_executor(None, self._recognizer.load)
            await self._send(ws, type="status", state="idle", message="模型就绪")

            while self._running:
                msg = await ws.receive_json()
                await self._on_message(ws, msg)
        except WebSocketDisconnect:
            pass
        except Exception:
            _log.exception("WebSocket 会话异常")
        finally:
            await self._cleanup()

    # ------------------------------------------------------------------
    # 消息路由
    # ------------------------------------------------------------------

    async def _on_message(self, ws: WebSocket, msg: dict) -> None:
        action = msg.get("action", "")

        if action == "start_capture":
            await self._start_session(ws)
        elif action == "process_frame":
            await self._process_frame(ws, msg.get("data", ""))
        elif action == "stop":
            await self._stop_session(ws)
        elif action == "recognize":
            await self._run_recognize(ws)
        elif action == "confirm_token":
            confirmed = self._recognizer.confirm_current()
            if confirmed:
                self._current_tokens.append(confirmed)
                await self._send(ws, type="tokens_append",
                                 data=[confirmed],
                                 total_tokens=self._current_tokens,
                                 count=len(self._current_tokens))
                self._translate_task = asyncio.create_task(
                    self._try_auto_translate(ws))
        elif action == "delete_token":
            idx = msg.get("index", -1)
            ok = self._recognizer.delete_token(idx)
            if ok and 0 <= idx < len(self._current_tokens):
                self._current_tokens.pop(idx)
            await self._send(ws, type="token_deleted", index=idx, success=ok,
                             tokens=self._current_tokens)
        elif action == "clear_tokens":
            self._recognizer.clear()
            self._current_tokens = []
            self._last_token_count = 0
            self._last_tokens_snapshot = ""
            await self._send(ws, type="tokens_clear")
        elif action == "confirm_translate":
            text = msg.get("text", "")
            await self._confirm_translate(ws, text)
        elif action == "generate_audio":
            await self._generate_audio(ws)
        elif action == "confirm_and_generate":
            text = msg.get("text", "")
            await self._confirm_and_generate(ws, text)
        elif action == "ping":
            await self._send(ws, type="pong")
        else:
            await self._send(ws, type="error", code="UNKNOWN_ACTION",
                             message=f"未知指令: {action}")

    # ------------------------------------------------------------------
    # 会话管理
    # ------------------------------------------------------------------

    async def _start_session(self, ws: WebSocket) -> None:
        """开始新识别会话（客户端已打开本地摄像头，服务端重置状态）。"""
        self._current_tokens = []
        self._current_sentence = ""
        self._current_history_id = None
        self._current_gender = "female"
        self._frame_idx = 0
        self._last_hands = []
        self._last_token_count = 0
        self._last_tokens_snapshot = ""
        self._recognizer.clear()

        if hasattr(self._sign._model, 'reset_session'):
            self._sign._model.reset_session()

        await self._send(ws, type="status", state="capturing",
                         message="摄像头已启动，正在实时采集...")
        await self._send(ws, type="tokens_clear")

    async def _stop_session(self, ws: WebSocket) -> None:
        self._recognizer.clear()
        self._current_tokens = []
        self._frame_idx = 0
        self._last_hands = []
        self._last_token_count = 0
        self._last_tokens_snapshot = ""
        await self._send(ws, type="detection", data=[])
        await self._send(ws, type="status", state="idle", message="已停止")

    # ------------------------------------------------------------------
    # 处理客户端发来的帧
    # ------------------------------------------------------------------

    async def _process_frame(self, ws: WebSocket, data: str) -> None:
        """
        客户端发来一帧 base64 JPEG → 跳帧检测 → 跳帧分类 → 实时猜测 + 自动翻译。

        优化：MediaPipe 每 2 帧跑 1 次，CSL 分类每 3 帧跑 1 次。
        """
        if not data:
            return

        _log = logging.getLogger(__name__)
        loop = asyncio.get_event_loop()
        self._frame_idx += 1

        try:
            # 解码 JPEG
            img_bytes = base64.b64decode(data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return

            detector = self._sign._model.detector
            if detector is None:
                return

            # ---- 1. MediaPipe 手部检测（跳帧）----
            run_detect = (self._detect_interval <= 1) or (self._frame_idx % self._detect_interval == 1)
            if run_detect:
                hands_data = await loop.run_in_executor(None, detector.detect, frame)
                self._last_hands = hands_data

                # 构建检测数据发给前端
                detection_payload = []
                for hand in hands_data:
                    item = {
                        "bbox": list(hand["bbox"]),
                        "handedness": hand.get("handedness", "Unknown"),
                        "confidence": float(hand.get("confidence", 0)),
                        "token": None,
                    }
                    if "landmarks_pixel" in hand:
                        lm = hand["landmarks_pixel"]
                        if hasattr(lm, "tolist"):
                            item["landmarks"] = lm.tolist()
                        else:
                            item["landmarks"] = [[float(p[0]), float(p[1])] for p in lm]
                    item["landmarks_count"] = len(item.get("landmarks", []))
                    detection_payload.append(item)
                await self._send(ws, type="detection", data=detection_payload)
            else:
                hands_data = self._last_hands

            # ---- 2. CSL 手语分类（跳帧 + 运动检测）----
            run_classify = (self._classify_interval <= 1) or (self._frame_idx % self._classify_interval == 1)
            if run_classify and hands_data:
                feature = build_hands_feature(hands_data)
                avg_conf = np.mean([h["confidence"] for h in hands_data])

                # 运动检测：手静止时不分类，避免误触发
                motion = 1.0  # 默认有运动
                if self._last_landmarks is not None:
                    motion = float(np.mean(np.abs(feature - self._last_landmarks)))
                self._last_landmarks = feature.copy()

                # 阈值 0.001：仅过滤完全静止，轻微呼吸抖动即可通过
                if motion > 0.001:
                    await loop.run_in_executor(
                        None, self._recognizer.classify_frame,
                        feature, avg_conf,
                    )

            # ---- 3. 实时猜测推送给前端 ----
            guess = self._recognizer.get_guess()
            if guess:
                await self._send(ws, type="guess_update", guess=guess)

            # ---- 4. 检测新确认的 Token ----
            recognizer_tokens = self._recognizer.get_tokens()
            if len(recognizer_tokens) > self._last_token_count:
                new_confirmed = recognizer_tokens[self._last_token_count:]
                self._last_token_count = len(recognizer_tokens)
                existing_set = set(self._current_tokens)
                fresh = [t for t in new_confirmed if t not in existing_set]
                if fresh:
                    self._current_tokens.extend(fresh)
                    await self._send(ws, type="tokens_append",
                                     data=fresh,
                                     total_tokens=self._current_tokens,
                                     count=len(self._current_tokens))
                    # 后台自动翻译
                    self._translate_task = asyncio.create_task(
                        self._try_auto_translate(ws))

        except Exception:
            _log.exception("process_frame 异常")

    # ------------------------------------------------------------------
    # 自动翻译（后台，不阻塞帧处理）
    # ------------------------------------------------------------------

    async def _try_auto_translate(self, ws: WebSocket) -> None:
        """Token 列表变化时自动连词成句，推送给前端。"""
        tokens = list(self._current_tokens)
        if not tokens:
            return
        snapshot = ",".join(tokens)
        if snapshot == self._last_tokens_snapshot:
            return
        self._last_tokens_snapshot = snapshot
        loop = asyncio.get_event_loop()
        try:
            sentence = await loop.run_in_executor(
                None, self.translate.translate, tokens,
            )
            await self._send(ws, type="auto_translate", data=sentence,
                             tokens=tokens, count=len(tokens))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 识别 + 翻译（用户主动触发）
    # ------------------------------------------------------------------

    async def _run_recognize(self, ws: WebSocket) -> None:
        loop = asyncio.get_event_loop()

        # 获取当前会话所有 Token
        all_tokens = self._recognizer.get_tokens()
        existing_set = set(self._current_tokens)
        for t in all_tokens:
            if t not in existing_set:
                self._current_tokens.append(t)
                existing_set.add(t)

        tokens = self._current_tokens

        if not tokens:
            await self._send(ws, type="error", code="NO_TOKENS",
                             message="还没有识别到手势，请先做手势")
            return

        await self._send(ws, type="tokens", data=tokens, count=len(tokens))
        await self._send(ws, type="status", state="recognizing",
                         message="正在识别手语...")

        # 性别识别
        self._current_gender = self.gender.detect(tokens)
        await self._send(ws, type="gender_result", gender=self._current_gender)

        # 翻译
        await self._send(ws, type="status", state="translating",
                         message="正在翻译内容...")
        try:
            sentence = await loop.run_in_executor(
                None, self.translate.translate, tokens,
            )
        except (ValueError, RuntimeError) as e:
            await self._send(ws, type="error", code="TRANSLATE_FAILED",
                             message=str(e))
            return

        self._current_sentence = sentence
        await self._send(ws, type="translation_done", data=sentence,
                         tokens=tokens, gender=self._current_gender)
        await self._send(ws, type="status", state="waiting_confirm",
                         message="请确认或修改翻译结果")

    # ------------------------------------------------------------------
    # 确认 / 生成语音（不变）
    # ------------------------------------------------------------------

    async def _confirm_translate(self, ws: WebSocket, text: str) -> None:
        if not text.strip():
            await self._send(ws, type="error", code="EMPTY_TEXT",
                             message="翻译文本不能为空")
            return
        self._current_sentence = text
        record_id = await asyncio.get_event_loop().run_in_executor(
            None, history_service.create_record,
            self._current_tokens, text,
        )
        self._current_history_id = record_id
        await self._send(ws, type="status", state="waiting_generate",
                         message="文本已确认，可生成语音")

    async def _confirm_and_generate(self, ws: WebSocket, text: str) -> None:
        if not text.strip():
            await self._send(ws, type="error", code="EMPTY_TEXT",
                             message="翻译文本不能为空")
            return
        self._current_sentence = text
        loop = asyncio.get_event_loop()
        await self._send(ws, type="status", state="generating_audio",
                         message="正在确认文本并生成语音...")
        record_id = await loop.run_in_executor(
            None, history_service.create_record,
            self._current_tokens, text,
        )
        self._current_history_id = record_id
        filename = f"{record_id}.wav"
        output_path = AUDIO_DIR / filename
        try:
            duration = await loop.run_in_executor(
                None, self.speech.synthesize_to_file,
                text, str(output_path), self._current_gender,
            )
        except RuntimeError as e:
            await self._send(ws, type="error", code="TTS_FAILED",
                             message=f"语音生成失败（记录已保存 ID={record_id}）: {e}")
            await self._send(ws, type="status", state="waiting_generate",
                             message="文本已确认，语音生成失败，可重试")
            return
        await loop.run_in_executor(
            None, history_service.update_audio_path,
            record_id, filename, duration,
        )
        await self._send(ws, type="audio_ready",
                         history_id=record_id, duration_sec=duration)
        await self._send(ws, type="status", state="audio_ready",
                         message="语音已生成")

    async def _generate_audio(self, ws: WebSocket) -> None:
        if self._current_history_id is None or not self._current_sentence:
            await self._send(ws, type="error", code="NOT_CONFIRMED",
                             message="请先确认翻译文本")
            return
        await self._send(ws, type="status", state="generating_audio",
                         message="正在生成语音...")
        loop = asyncio.get_event_loop()
        filename = f"{self._current_history_id}.wav"
        output_path = AUDIO_DIR / filename
        try:
            duration = await loop.run_in_executor(
                None, self.speech.synthesize_to_file,
                self._current_sentence, str(output_path), self._current_gender,
            )
        except RuntimeError as e:
            await self._send(ws, type="error", code="TTS_FAILED", message=str(e))
            return
        await loop.run_in_executor(
            None, history_service.update_audio_path,
            self._current_history_id, filename, duration,
        )
        await self._send(ws, type="audio_ready",
                         history_id=self._current_history_id,
                         duration_sec=duration)
        await self._send(ws, type="status", state="audio_ready",
                         message="语音已生成")

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    async def _send(self, ws: WebSocket, **kwargs) -> None:
        try:
            await ws.send_json(kwargs)
        except WebSocketDisconnect:
            self._running = False

    async def _cleanup(self) -> None:
        self._running = False
        self._recognizer.clear()
        if hasattr(self._sign._model, 'reset_session'):
            self._sign._model.reset_session()
