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
import os

# 【知识点：__file__】config.py 在 src/ 下，parent.parent 回到项目根目录
ROOT = Path(__file__).parent.parent

# 【知识点：HuggingFace 缓存机制】
# HF_HOME 和 HF_HUB_CACHE 控制 HuggingFace 下载模型权重的存放位置。
# 默认在用户目录 ~/.cache/huggingface，这里改到项目本地，方便打包和迁移。
SRC = ROOT / "src"
MODEL_CACHE_DIR = SRC / "models" / "cache"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE_DIR / "hub")
# 国内用户可设置 HF_ENDPOINT=https://hf-mirror.com 加速下载
# os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 离线模式：仅使用本地缓存。设置 HF_HUB_OFFLINE=1 可跳过在线检查
# os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ---- 模型路径 ----
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
TTS_RATE = 200         # 语速，默认200词/分钟

# ---- 服务器 ----
# 【知识点：FastAPI + uvicorn】FastAPI 是 Python 异步 Web 框架，uvicorn 是 ASGI 服务器
# 【知识点：SSL/TLS】自签名证书实现 HTTPS，浏览器会警告但数据加密传输
# 部署时通过环境变量覆盖：HOST / PORT / SSL_KEYFILE / SSL_CERTFILE
# SSL_KEYFILE 为空字符串时禁用 HTTPS（配合 Nginx 反代使用）
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
_ssl_key = os.environ.get("SSL_KEYFILE")  # None=未设置用默认值, ""=禁用HTTPS
_ssl_cert = os.environ.get("SSL_CERTFILE")
SSL_KEYFILE = Path(_ssl_key) if _ssl_key else (ROOT / "certs" / "key.pem")
SSL_CERTFILE = Path(_ssl_cert) if _ssl_cert else (ROOT / "certs" / "cert.pem")

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

# ---- Token 相邻黑名单 ----
# (前一个词, 后一个词) 禁止相邻出现，过滤不合逻辑的识别结果
# A=同词重复 C="很"后只跟少数形容词 E=其他不通顺
CSL_TOKEN_BLACKLIST: set[tuple[str, str]] = {
    # === A. 同词重复（22类）===
    ("我", "我"), ("你", "你"), ("很", "很"), ("去", "去"), ("吃", "吃"),
    ("在", "在"), ("会", "会"), ("喜欢", "喜欢"), ("上次", "上次"),
    ("面包", "面包"), ("手语", "手语"), ("大家", "大家"), ("一起", "一起"),
    ("为什么", "为什么"), ("对不起", "对不起"), ("谢谢", "谢谢"),
    ("开心", "开心"), ("祝", "祝"), ("需要", "需要"),
    ("帮助", "帮助"), ("请", "请"), ("问", "问"),
    # === C. "很" 后只能跟 [喜欢/对不起/谢谢/开心] ===
    ("很", "你"), ("很", "我"), ("很", "在"), ("很", "去"),
    ("很", "上次"), ("很", "会"), ("很", "大家"),
    ("很", "面包"), ("很", "一起"), ("很", "吃"),
    ("很", "为什么"), ("很", "祝"), ("很", "需要"),
    ("很", "帮助"), ("很", "请"), ("很", "问"), ("很", "手语"),
    # === E. 其他不通顺 ===
    ("面包", "手语"),
}
