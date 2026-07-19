"""
REST API 路由。

支持两套流程：
1. 摄像头实时流（WebSocket）
2. 视频文件上传（REST + 后续 WebSocket/HTTP 交互）
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from src.models import get_sign_language_model
from src.services.sign_service import SignService
from src.services.translate_service import TranslateService
from src.services.speech_service import SpeechService
from src.services.video_service import VideoService
from src.services import history_service
from src.config import AUDIO_DIR, TEMP_DIR, TRANSLATION_MODE

router = APIRouter(prefix="/api")

_video_service: VideoService | None = None


def get_video_service() -> VideoService:
    global _video_service
    if _video_service is None:
        _video_service = VideoService()
    return _video_service


# ------------------------------------------------------------------
# 请求/响应模型
# ------------------------------------------------------------------

class TranslateRequest(BaseModel):
    words: list[str]

class TTSRequest(BaseModel):
    text: str
    gender: str = "female"

class VideoConfirmRequest(BaseModel):
    tokens: list[str]
    text: str


# ------------------------------------------------------------------
# 系统端点
# ------------------------------------------------------------------

@router.get("/health")
async def health():
    return {"status": "ok", "service": "基于Vision Transformer的手语识别生成语音系统"}

@router.get("/status")
async def status():
    slm = get_sign_language_model()
    translate_svc = TranslateService()
    return {
        "sign_language_model": {
            "type": type(slm).__name__,
            "loaded": slm.is_loaded(),
        },
        "text_translate_model": {
            "type": "Qwen2TranslateModel" if TRANSLATION_MODE == "qwen" else "MockTranslateModel",
            "loaded": translate_svc.is_loaded(),
            "mode": translate_svc.current_mode,
        },
    }

@router.post("/translate")
async def translate(req: TranslateRequest):
    """手动输入词汇列表 → 翻译。"""
    if not req.words:
        raise HTTPException(status_code=400, detail="words 不能为空")
    try:
        service = TranslateService()
        sentence = service.translate(req.words)
        return {"input": req.words, "output": sentence}
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tts")
async def tts(req: TTSRequest):
    """文本 → WAV 音频。"""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text 不能为空")
    try:
        service = SpeechService()
        audio = service.synthesize(req.text, gender=req.gender)
        return Response(content=audio, media_type="audio/wav")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# 视频文件处理端点
# ------------------------------------------------------------------

@router.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    """上传手语视频 → 抽帧 → 识别 → 翻译。"""
    allowed_extensions = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".webm"}
    ext = ("." + file.filename.rsplit(".", 1)[-1].lower()) if file.filename and "." in file.filename else ""
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}，支持: {', '.join(allowed_extensions)}",
        )

    video_path = TEMP_DIR / f"upload_{file.filename}"
    try:
        content = await file.read()
        video_path.write_bytes(content)

        video_svc = get_video_service()
        frame_paths = video_svc.extract_frames(str(video_path))

        if not frame_paths:
            raise HTTPException(status_code=400, detail="未能从视频中提取到有效帧")

        sign_model = get_sign_language_model()
        sign_svc = SignService(sign_model)
        tokens = sign_svc.process_frame_files(frame_paths)

        translate_svc = TranslateService()
        sentence = translate_svc.translate(tokens)

        video_svc.clear_frames()

        return {
            "tokens": tokens,
            "sentence": sentence,
            "frame_count": len(frame_paths),
        }

    except HTTPException:
        raise
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=500, detail=f"视频处理失败: {e}")
    finally:
        if video_path.exists():
            video_path.unlink(missing_ok=True)


# ------------------------------------------------------------------
# 视频流程：保存结果 + 生成音频
# ------------------------------------------------------------------

@router.post("/save-video-result")
async def save_video_result(req: VideoConfirmRequest):
    """保存视频翻译结果到历史记录。"""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text 不能为空")
    if not req.tokens:
        raise HTTPException(status_code=400, detail="tokens 不能为空")

    record_id = history_service.create_record(req.tokens, req.text)
    return {"history_id": record_id, "text": req.text}

@router.post("/confirm-video")
async def confirm_video(req: VideoConfirmRequest):
    """统一确认：保存翻译结果 + 生成语音 + 返回完整记录。"""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text 不能为空")
    if not req.tokens:
        raise HTTPException(status_code=400, detail="tokens 不能为空")

    record_id = history_service.create_record(req.tokens, req.text)

    try:
        speech = SpeechService()
        filename = f"{record_id}.wav"
        output_path = AUDIO_DIR / filename
        duration = speech.synthesize_to_file(req.text, str(output_path))

        history_service.update_audio_path(record_id, filename, duration)

        return {
            "history_id": record_id,
            "text": req.text,
            "duration_sec": duration,
            "audio_url": f"/api/audio/{filename}",
        }
    except RuntimeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"语音生成失败（记录已保存 ID={record_id}）: {e}",
        )

@router.post("/generate-audio/{record_id}")
async def generate_audio(record_id: int):
    """为已确认的历史记录生成语音文件。"""
    record = history_service.get_record(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="历史记录不存在")
    if not record["translated_text"]:
        raise HTTPException(status_code=400, detail="翻译文本为空")

    try:
        speech = SpeechService()
        filename = f"{record_id}.wav"
        output_path = AUDIO_DIR / filename
        duration = speech.synthesize_to_file(record["translated_text"], str(output_path))

        history_service.update_audio_path(record_id, filename, duration)

        return {
            "history_id": record_id,
            "duration_sec": duration,
            "audio_url": f"/api/audio/{filename}",
        }
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"语音生成失败: {e}")


# ------------------------------------------------------------------
# 历史记录端点
# ------------------------------------------------------------------

@router.get("/history")
async def list_history():
    records = history_service.get_all_records()
    return records

@router.delete("/history/{record_id}")
async def delete_history(record_id: int):
    ok = history_service.delete_record(record_id)
    if not ok:
        raise HTTPException(status_code=404, detail="记录不存在")
    return {"status": "deleted", "id": record_id}

@router.get("/audio/{filename}")
async def serve_audio(filename: str):
    audio_path = AUDIO_DIR / filename
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="音频文件不存在")
    return FileResponse(str(audio_path), media_type="audio/wav")
