"""
【知识点：配置管理模式】
将所有可调参数集中在单一文件，其他模块通过 import 使用。
好处：改参数只需改一处，不会出现在代码各处散落魔法数字的问题。

涉及知识点：
  - pathlib.Path: Python 3.4+ 的面向对象路径库，跨平台处理文件路径
  - os.environ.setdefault: 只在环境变量未设置时才赋值，不会覆盖已有值
  - 配置常量命名约定：全大写 + 下划线（如 CAMERA_FPS）
"""
from pathlib import Path

# 【知识点：__file__】config.py 在 src/ 下，parent.parent 回到项目根目录
ROOT = Path(__file__).parent.parent

# 【知识点：HuggingFace 缓存机制】
# HF_HOME 和 HF_HUB_CACHE 控制 HuggingFace 下载模型权重的存放位置。
# 默认在用户目录 ~/.cache/huggingface，这里改到项目本地，方便打包和迁移。
SRC = ROOT / "src"
MODEL_CACHE_DIR = SRC / "models" / "cache"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

import os
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE_DIR / "hub")
# 国内用户使用 hf-mirror.com 镜像，避免下载超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 离线模式：仅使用本地缓存，不尝试连接远程
os.environ["HF_HUB_OFFLINE"] = "1"

# ---- 模型路径 ----
LORA_PATH = SRC / "models" / "text_model" / "lora_word2sent"
# Qwen2-0.5B 本地缓存路径，避免每次启动尝试连接 HuggingFace
_TEXT_MODEL_SNAPSHOT = (
    MODEL_CACHE_DIR / "hub" / "models--Qwen--Qwen2-0.5B-Instruct"
    / "snapshots" / "c540970f9e29518b1d8f06ab8b24cba66ad77b6d"
)
TEXT_MODEL_NAME = str(_TEXT_MODEL_SNAPSHOT) if _TEXT_MODEL_SNAPSHOT.exists() else "Qwen/Qwen2-0.5B-Instruct"

# ---- 摄像头 ----
# 【知识点：cv2.VideoCapture】OpenCV 视频采集，0 = 默认摄像头
CAMERA_INDEX = 0
# 【知识点：帧率 FPS】Frames Per Second，越高画面越流畅但计算量越大
CAMERA_FPS = 12
# 【知识点：环形缓冲区】固定大小的队列，新的进来旧的丢弃，防止内存爆炸
FRAME_BUFFER_SIZE = 60

# ---- 手语识别 ----
# 【知识点：自动识别 vs 手动触发】每隔 N 秒自动识别一次，0 = 禁用（仅手动按钮触发）
SIGN_AUTO_INTERVAL = 3.0
SIGN_MODEL_PATH = SRC / "models" / "sign_language_model" / "pretrained" / "hand_gesture_19"

CSL_MODEL_PATH = SRC / "models" / "sign_language_model" / "pretrained" / "csl_model.pt"
CSL_VOCABULARY_PATH = SRC / "models" / "sign_language_model" / "pretrained" / "csl_vocabulary.json"
# 【知识点：特征维度】每手 21 个关键点 × 3 坐标(x,y,z) = 63 维，双手 ×2 = 126 维
CSL_INPUT_DIM: int = 126

# ---- MediaPipe 检测参数 ----
# 【知识点：置信度阈值】值越高越严格（漏检多、误检少），越低越宽松
MEDIAPIPE_MIN_DETECTION = 0.3   # 降低阈值，减少检测框消失
MEDIAPIPE_MIN_TRACKING = 0.3
MEDIAPIPE_MAX_HANDS = 2

# ---- 实时识别参数 ----
REALTIME_RECOGNIZE_INTERVAL = 12  # 每隔12帧做一次识别推断，减少计算压力
# 【知识点：置信度阈值 + 稳定性过滤】两层过滤：先过滤低置信度预测，再要求连续N帧一致
CSL_CONFIDENCE_THRESHOLD = 0.55   # 提高阈值，过滤低置信度的乱猜
CSL_STABILITY_THRESHOLD = 5        # 连续5帧一致才确认，减少跳变
CSL_COOLDOWN_FRAMES = 30           # 输出后冷却30帧，防止重复输出

# ---- 语音合成 (TTS) ----
# 【知识点：pyttsx3】离线 TTS 引擎，调用操作系统内置语音（Windows SAPI5 / macOS NSSpeech）
TTS_VOICE_ID = 0       # 0=中文语音(Huihui), 1=英文语音(Zira)
TTS_RATE = 200         # 语速，默认200词/分钟

# ---- 服务器 ----
# 【知识点：FastAPI + uvicorn】FastAPI 是 Python 异步 Web 框架，uvicorn 是 ASGI 服务器
# 【知识点：SSL/TLS】自签名证书实现 HTTPS，浏览器会警告但数据加密传输
HOST = "0.0.0.0"       # 0.0.0.0 表示监听所有网络接口，局域网可访问
PORT = 8000
SSL_KEYFILE = ROOT / "certs" / "key.pem"
SSL_CERTFILE = ROOT / "certs" / "cert.pem"

# ---- 临时文件 ----
TEMP_DIR = ROOT / "tmp"
TEMP_DIR.mkdir(exist_ok=True)

# ---- 文本翻译模式 ----
# 【知识点：降级策略】auto 模式下：优先 Qwen2 → 失败则降级到 Mock 映射表
# qwen = 强制 Qwen2（失败报错），mock = 仅用映射表（不加载模型）
TRANSLATION_MODE: str = "qwen"

# ---- 音频 ----
AUDIO_DIR = ROOT / "audio"
AUDIO_DIR.mkdir(exist_ok=True)

# ---- ViT 视觉特征提取 ----
# 【知识点：Vision Transformer (ViT)】将图像切成 16×16 patch，用 Transformer 编码
# USE_VIT=False 时只用 MediaPipe 关键点（126维）
# USE_VIT=True  时拼接 ViT 视觉特征（126+768=894维），多模态融合
USE_VIT: bool = False  # 默认关闭，首次启用需下载 ~330MB 权重
