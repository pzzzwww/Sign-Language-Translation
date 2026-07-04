"""
FastAPI 应用入口。
"""

from __future__ import annotations

import atexit
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.api.routes import router as api_router
from src.websocket.handler import StreamHandler
from src.config import AUDIO_DIR, ROOT
from src.services.sign_service import SignService
from src.services.translate_service import TranslateService
from src.services.speech_service import SpeechService
from src.services.database import init_db
from src.models import get_sign_language_model

# ------------------------------------------------------------------
# 初始化数据库
# ------------------------------------------------------------------
init_db()

# ------------------------------------------------------------------
# 全局服务实例（单例）
# ------------------------------------------------------------------
_sign_model = get_sign_language_model()
_sign_model.load()  # 启动时预加载模型，避免首次连接等待
sign_service = SignService(_sign_model)
translate_service = TranslateService()
translate_service._get_model()  # 启动时预加载翻译模型，避免首次请求等待/崩溃
speech_service = SpeechService()


def _cleanup() -> None:
    sign_service.stop_camera()
    _sign_model.unload()
    translate_service.unload()


atexit.register(_cleanup)

# ------------------------------------------------------------------
# FastAPI 应用
# ------------------------------------------------------------------
app = FastAPI(
    title="基于Transformer的手语识别生成语音系统",
    description="实时手语识别与语音合成系统",
    version="0.2.0",
)

app.include_router(api_router)

# 挂载音频目录（用于历史记录播放）
if AUDIO_DIR.exists():
    app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")

# 挂载前端静态文件
STATIC_DIR = ROOT / "frontend" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ------------------------------------------------------------------
# WebSocket 端点
# ------------------------------------------------------------------

@app.websocket("/ws/stream")#FastAPI带WebSocket类
async def websocket_endpoint(ws: WebSocket):#FastAPI框架把WebSocket实例对象传进来
    handler = StreamHandler(sign_service, translate_service, speech_service)
    await handler.handle(ws)


# ------------------------------------------------------------------
# 前端页面
# ------------------------------------------------------------------

@app.get("/")
async def index():
    html_path = ROOT / "frontend" / "index.html"
    return FileResponse(str(html_path))


# ------------------------------------------------------------------
# 启动入口
# ------------------------------------------------------------------

def main() -> None:
    """CLI 入口：启动 FastAPI 服务。"""
    import os
    import uvicorn
    from src.config import HOST, PORT, SSL_KEYFILE, SSL_CERTFILE

    _no_ssl_env = os.environ.get("NO_SSL", "").strip() in ("1", "true", "yes")
    _ssl_ok = (str(SSL_KEYFILE) and str(SSL_CERTFILE)
               and SSL_KEYFILE.exists() and SSL_CERTFILE.exists())
    if _no_ssl_env or not _ssl_ok:
        print(f"\n{'='*60}")
        print(f"  基于Transformer的手语识别生成语音系统 v0.3")
        print(f"  访问地址: http://localhost:{PORT}")
        print(f"  SSL: 已关闭")
        print(f"{'='*60}\n")
        uvicorn.run(app, host=HOST, port=PORT, log_level="info")
    else:
        print(f"\n{'='*60}")
        print(f"  基于Transformer的手语识别生成语音系统 v0.3")
        print(f"  访问地址: https://localhost:{PORT}")
        print(f"{'='*60}\n")
        uvicorn.run(app, host=HOST, port=PORT, log_level="info",
                    ssl_keyfile=str(SSL_KEYFILE), ssl_certfile=str(SSL_CERTFILE))


if __name__ == "__main__":
    main()
