"""
WebSocket 连接管理器。
每个客户端使用自己的摄像头 → 发帧到服务端 → 服务端处理 → 返回结果。
每个 WebSocket 连接维护独立的识别会话。
"""

from __future__ import annotations

import asyncio
import base64
import logging
import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from src.services.translate_service import TranslateService
from src.services.speech_service import SpeechService
from src.services import history_service
from src.models.sign_language_model.csl_recognizer import CSLRecognizer
from src.models.sign_language_model.real_recognizer import build_hands_feature
from src.config import AUDIO_DIR, CSL_MODEL_PATH


class StreamHandler:
    """每个 WebSocket 连接一个实例，独立识别会话。"""

    def __init__(self,sign,translate: TranslateService,speech: SpeechService) -> None:
        self._sign = sign
        self.translate = translate
        self.speech = speech
        self._ws: WebSocket | None = None
        self._running = False
        # 当前会话状态
        self._current_tokens: list[str] = []
        self._current_sentence: str = ""
        self._current_history_id: int | None = None
        self._current_gender: str = "female"
        # 跳帧优化 + 自动翻译
        self._frame_idx = 0
        self._last_hands: list[dict] = []
        self._last_token_count = 0       # 跟踪 recognizer 已确认 token 数量
        self._last_tokens_snapshot = ""  # 自动翻译去重
        self._translate_task: asyncio.Task | None = None
        # 每个连接独立的 CSL 识别器（加载训练权重，追踪 Token 序列）
        self._recognizer = CSLRecognizer(
            model_path=CSL_MODEL_PATH,
            confidence_threshold=0.35,
            stability_threshold=2,
            cooldown_frames=8,
        )
    #入口
    async def handle(self, ws: WebSocket) -> None:#同意连接，把模型加载好，通知前端"我准备好了"，客户端每发来一帧，处理一帧
        _log = logging.getLogger(__name__)
        try:
            await ws.accept()#同意连接需要时间，cpu去办别的事情
        except WebSocketDisconnect:
            return
        self._ws = ws
        self._running = True

        try:
            loop = asyncio.get_event_loop()
            model = self._sign.model
            if not model.is_loaded():
                await self._send(ws, type="status", state="loading_model",
                                 message="正在加载手语识别模型...")
                await loop.run_in_executor(None, model.load)
            await self._send(ws, type="status", state="loading_model",
                             message="正在加载 CSL 识别器...")
            await loop.run_in_executor(None, self._recognizer.load)
            await self._send(ws, type="status", state="idle", message="模型就绪")

            while self._running: #客户端发一帧 → 服务端处理一帧 → 等下一帧 → 再处理。一直循环到客户端断开或出错。
                msg = await ws.receive_json()# 等客户端发来一帧
                await self._on_message(ws, msg) # 处理这一帧
        except WebSocketDisconnect:
            pass
        except Exception:
            _log.exception("WebSocket 会话异常")
        finally:
            await self._cleanup()

    #分发处理
    async def _on_message(self, ws: WebSocket, msg: dict) -> None:#客户端发的每条消息里带一个 action 字段，这个函数看 action 是什么，调对应的处理逻辑。
        action = msg.get("action", "")

        if action == "start_capture":
            await self._start_session(ws)
        elif action == "process_frame": #处理一帧画面
            await self._process_frame(ws, msg.get("data", ""))
        elif action == "stop":
            await self._stop_session(ws)
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
        elif action == "generate_audio":#生成语音
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

        self._sign.reset_session()

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


    #处理客户端发来的帧
    async def _process_frame(self, ws: WebSocket, data: str) -> None:
        """
        客户端发来的：一帧=字符串
        服务端做的事：
        ① base64 解码 → 二进制 → numpy → 图片
        ② MediaPipe 提取 21 个手部关键点坐标
        ③ 关键点坐标发给前端（前端按照关键点范围自己画框画点）
        ④ 关键点坐标拼成 126 维向量 → 模型分类 → 手势文字(攒够12个126维向量送入模型推理一次)
        ⑤ 新确认的多个手势文字 → 加入当前 token 列表
        ⑥ token 列表有变化 → 后台自动调 Qwen2 翻译
        ⑦ 翻译结果 → 推送给前端显示

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

            detector = self._sign.model.detector
            if detector is None:
                return

            # ---- 1. MediaPipe 手部检测 ----
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

            # ---- 2. CSL 手语分类 ----
            if hands_data:
                feature = build_hands_feature(hands_data)
                await loop.run_in_executor(
                    None, self._recognizer.classify_frame,
                    feature,
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


    # 自动翻译（后台，不阻塞帧处理）
    async def _try_auto_translate(self, ws: WebSocket) -> None:#入参:FastAPI的WebSocket连接对象，用于向这个客户端推送消息
        """Token 列表变化时自动连词成句，推送给前端。"""
        tokens = list(self._current_tokens)#取Token列表，把当前累积的Token复制出来
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
            )#把翻译任务丢到线程池执行，翻译结果，推送给前端
            await self._send(ws, type="auto_translate", data=sentence,tokens=tokens, count=len(tokens))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 确认 / 生成语音
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
        self._sign.reset_session()
