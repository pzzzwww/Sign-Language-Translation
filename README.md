# 基于Vision Transformer的手语识别生成语音系统

> 实时手语识别与语音合成系统 — 让无声的表达被听见。

## 项目简介

**基于Vision Transformer的手语识别生成语音系统** 是一个端到端的实时手语翻译与语音合成平台。用户可通过**摄像头实时采集**或**上传视频文件**两种方式输入手语，系统自动识别手势词汇、翻译为自然中文语句，并合成为可播放的语音。所有翻译记录自动保存，支持历史回放。

### 核心流程

```
摄像头/视频 → MediaPipe 手部关键点检测 → ViT-B/16 视觉特征提取
    → CSL Transformer 时序分类 → Token 序列
    → Qwen2-0.5B + LoRA 翻译 → 自然中文语句
    → pyttsx3 TTS → 语音播放
```

### 特性

- **双模式输入**：摄像头实时流（WebSocket）+ 视频文件上传（REST API）
- **实时帧推流**：WebSocket 推送摄像头 JPEG 帧，前端 Canvas 渲染
- **多模态手语识别**：MediaPipe 21点手部关键点 + ViT-B/16 768维视觉特征 → Transformer Encoder 时序分类
- **大模型翻译**：Qwen2-0.5B-Instruct + LoRA 微调，词汇序列 → 通顺中文句子
- **分步流水线**：识别 → 翻译 → 用户确认/编辑 → 生成语音，每步可独立控制
- **自动降级**：模型权重缺失时自动降级为启发式规则，零依赖可用
- **翻译历史**：SQLite 持久化存储，支持查询、回放、删除
- **离线 TTS**：pyttsx3 本地语音合成，无需网络

---

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI + Uvicorn | 异步 REST + WebSocket |
| 实时通信 | WebSocket | 摄像头帧推流，双向消息 |
| 手部检测 | MediaPipe | 21点手部关键点实时提取 |
| 视觉编码 | ViT-B/16 | 768维手部区域视觉特征（可选） |
| 时序分类 | CSL Transformer Encoder | 4层 8头自注意力 + 可学习位置编码 |
| 文本翻译 | Qwen2-0.5B-Instruct + LoRA | PEFT 微调，chat template 推理 |
| 语音合成 | pyttsx3（离线 TTS） | Windows SAPI5，支持性别选择 |
| 视频处理 | OpenCV + MoviePy | 抽帧、预处理 |
| 数据存储 | SQLite | 翻译历史持久化 |
| 前端 | 原生 HTML/CSS/JS | 零框架依赖，WebSocket 双向通信 |

### 架构设计亮点

- **Strategy 模式**：`SignLanguageModel` / `TextTranslateModel` 抽象接口，工厂函数创建实例，替换模型零业务代码改动
- **双入口架构**：WebSocket 实时流 + REST API 文件处理，共享同一服务层
- **配置集中管理**：`src/config.py` 统一控制所有开关（`USE_VIT`、`TRANSLATION_MODE` 等）
- **降级容错**：模型不可用时自动降级为启发式规则，保证系统可用性

---

## 项目结构

```
├── src/
│   ├── backend/
│   │   └── main.py              # FastAPI 应用入口
│   ├── api/
│   │   └── routes.py            # REST API 路由（/api）
│   ├── websocket/
│   │   └── handler.py           # WebSocket 连接管理 + 逐帧识别
│   ├── services/
│   │   ├── sign_service.py      # 手语识别服务（摄像头/视频帧处理）
│   │   ├── translate_service.py # Token → 中文翻译服务
│   │   ├── speech_service.py    # TTS 语音合成服务
│   │   ├── video_service.py     # 视频抽帧服务
│   │   ├── history_service.py   # 翻译历史 CRUD
│   │   └── database.py          # SQLite 数据库初始化
│   ├── models/
│   │   ├── sign_language_model/
│   │   │   ├── csl_recognizer.py      # CSL Transformer 时序分类器（主力）
│   │   │   ├── mediapipe_detector.py  # MediaPipe 手部关键点提取
│   │   │   ├── vit_encoder.py         # ViT-B/16 视觉特征编码
│   │   │   ├── real_recognizer.py     # 三合一识别器（MediaPipe + ViT + CSL）
│   │   │   ├── recognizer.py          # Hand-Gesture-19 识别器（旧版保留）
│   │   │   ├── placeholder_model.py   # 占位模型（降级用）
│   │   │   └── pretrained/            # CSL 训练权重 + MediaPipe 模型
│   │   └── text_model/
│   │       ├── qwen2_lora_model.py    # Qwen2 + LoRA 翻译模型
│   │       ├── mock_model.py          # 轻量映射表翻译（零模型依赖）
│   │       └── lora_word2sent/        # LoRA 适配器权重
│   ├── interfaces/
│   │   ├── sign_language_model.py     # SignLanguageModel 抽象接口
│   │   └── text_translate_model.py    # TextTranslateModel 抽象接口
│   ├── utils/
│   │   └── frame_processor.py         # 图像预处理工具
│   └── config.py                      # 集中配置
├── scripts/
│   ├── collect_data.py          # 手势数据采集工具
│   ├── train_csl.py             # CSL Transformer 训练脚本
│   ├── export_onnx.py           # ONNX 模型导出 + 推理基准测试
│   └── gradio_app.py            # Gradio 交互式 Demo
├── frontend/
│   ├── index.html               # 前端页面
│   └── static/
│       ├── css/style.css        # 样式表
│       └── js/main.js           # 前端交互逻辑
├── tests/
├── requirements.txt
└── README.md
```

---

## 快速启动

### 环境要求

- Python 3.10+，conda 虚拟环境
- PyTorch 2.5+（CUDA 可选）
- 摄像头（实时模式需要）

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 采集手势数据 + 训练（首次使用）

```bash
# 采集手势关键点数据（摄像头前比划，每手势 20 段）
python scripts/collect_data.py

# 训练 CSL Transformer 模型
python scripts/train_csl.py
```

### 3. 启动服务

```bash
python -m src.backend.main
```

或使用热重载：

```bash
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 打开界面

浏览器访问 **https://localhost:8000**（自签名证书需手动信任）

---

## 使用指南

### 摄像头模式（实时识别）

1. 打开页面，点击"**启动摄像头**"
2. 摄像头画面出现在 Canvas 区域
3. 在摄像头前做出手语手势
4. 点击"**开始识别**"（或开启自动识别模式）
5. 系统识别手语并显示 Token 列表和翻译结果
6. 可编辑翻译文本后点击"**确认文本**"
7. 点击"**生成语音**"，播放合成音频

### 视频模式（上传手语视频）

1. 切换到"**视频上传**"标签页
2. 选择手语视频文件（支持 mp4 / mov / avi / mkv / flv / webm）
3. 点击"**处理视频**"，系统自动抽帧 → 识别 → 翻译
4. 查看翻译结果，可编辑文本
5. 点击"**确认并生成语音**"，一步完成保存和语音合成

---

## API 概览

### REST API（`/api`）

| 方法 | 端点 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 健康检查 |
| `GET` | `/api/status` | 模型加载状态 |
| `POST` | `/api/translate` | 词汇列表 → 翻译句子 |
| `POST` | `/api/tts` | 文本 → WAV 音频流 |
| `POST` | `/api/process-video` | 上传视频 → 抽帧识别翻译 |
| `POST` | `/api/save-video-result` | 保存翻译结果到历史 |
| `POST` | `/api/confirm-video` | 组合：保存 + 生成语音一步完成 |
| `POST` | `/api/generate-audio/{id}` | 为已有记录生成语音 |
| `GET` | `/api/history` | 获取翻译历史列表 |
| `DELETE` | `/api/history/{id}` | 删除历史记录及音频 |
| `GET` | `/api/audio/{filename}` | 获取音频文件 |

### WebSocket（`/ws/stream`）

| action | 说明 |
|--------|------|
| `start_capture` | 启动摄像头推流 |
| `stop` | 停止摄像头 |
| `recognize` | 执行识别 + 翻译 |
| `process_video_frames` | 处理视频帧文件 |
| `confirm_translate` | 确认翻译文本 |
| `generate_audio` | 生成语音 |
| `set_auto` | 设置自动识别间隔 |
| `ping` | 心跳检测 |

---

## 模型架构详解

### 手语识别管线

```
帧输入 → MediaPipeHandDetector（21关键点 × 2手 = 126维）
       → ViTFeatureExtractor（ViT-B/16，768维手部视觉特征，可选）
       → CSLRecognizer（Transformer Encoder，4层 8头自注意力）
       → Token 序列输出
```

- **输入维度可配置**：单手 63 维，双手 126 维（`CSL_INPUT_DIM`）
- **ViT 融合开关**：`USE_VIT` 控制是否启用多模态融合
- **训练时自动学习**：类别权重 + 过采样处理不均衡，输出混淆矩阵

### 文本翻译管线

```
Token 序列 → Qwen2LoRAModel（Qwen2-0.5B + LoRA rank=8）
           → MockTranslateModel（降级备选，映射表翻译）
           → 自然中文句子
```

- **翻译模式**：`TRANSLATION_MODE = "auto" | "qwen" | "mock"`
- **LoRA 高效微调**：仅训练 adapter 权重（~6MB），基座模型冻结
- **Chat Template 推理**：使用 Qwen2 官方 chat template 格式

---

## 常见问题

**Q: 启动时报 CUDA 相关错误？**
A: 检查 PyTorch 是否匹配 CUDA 版本，模型会自动 fallback 到 CPU。

**Q: 摄像头打不开？**
A: 修改 `src/config.py` 中 `CAMERA_INDEX`，Windows 可能为 0 或 1。

**Q: 模型加载很慢？**
A: 首次启动需下载 Qwen2-0.5B-Instruct（约 1GB），后续使用缓存。翻译模式设为 `mock` 可跳过。

**Q: 手语识别没反应？**
A: 检查是否已训练 CSL 模型（`scripts/train_csl.py`）。未训练时自动降级为启发式规则。

**Q: 语音合成没有声音？**
A: pyttsx3 依赖系统 TTS 引擎。Windows 使用 SAPI5，Linux 需安装 `espeak`。

---

## License

内部项目，仅供学习和研究使用。
