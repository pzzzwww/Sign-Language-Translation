# CLAUDE.md

## 环境

项目使用 conda 虚拟环境 `torch`。所有 Python 命令、pip 安装、依赖检查必须使用该环境的 Python，不要使用系统 Python。

```bash
# Python 路径
C:/Users/yng/.conda/envs/torch/python.exe

# 当前已安装依赖:
#   torch 2.5.1, transformers 5.9.0, fastapi 0.136.3
#   mediapipe 0.10.7, cv2 4.9.0, numpy 1.26.4
#   gradio 未安装（仅 Gradio demo 需要）
```

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于Vision Transformer的手语识别生成语音系统 — 实时手语识别与语音合成平台。

核心流程：手势输入 → MediaPipe 手部关键点检测 → [ViT-B/16 视觉特征 (可选)] → Transformer Encoder 时序分类 → Token 输出 → Qwen2 翻译为自然句子 → pyttsx3 / CosyVoice2 TTS 语音合成。

## 部署与演示

```bash
# Gradio 演示（一键启动网页交互界面）
python gradio_app.py --share

# ONNX 导出（模型加速部署）
python scripts/export_onnx.py --mode fp32

# Docker 部署
docker-compose up -d
```

## 启动与测试

```bash
# 安装依赖
pip install -r requirements.txt

# ===== 数据采集 + 训练（首次使用必须）=====
# 1. 采集手势数据（摄像头前比划，每手势 20 段）
python scripts/collect_data.py
# 2. 训练 CSLTransformer 模型（GPU 自动检测）
python scripts/train_csl.py
# ===========================================

# 启动服务（HTTPS localhost:8000）
python -m src.backend.main

# 或使用 uvicorn 热重载
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload

# Gradio 交互式 demo
python gradio_app.py --share

# ONNX 模型导出
python scripts/export_onnx.py --mode fp32
```

## 架构

### 模型分层（Strategy 模式）

两个抽象接口定义在 `src/interfaces/`，业务代码只通过工厂函数获取实例：

- `src/interfaces/sign_language_model.py` — `SignLanguageModel` 抽象基类：`load() / predict(frames) / unload() / is_loaded()`
- `src/interfaces/text_translate_model.py` — `TextTranslateModel` 抽象基类：`load() / translate(words) / unload() / is_loaded()`

`src/models/__init__.py` 提供工厂函数 `get_sign_language_model()` / `get_text_translate_model()` 和 `register_*()` 注册函数。替换模型实现时只需调用 register 函数，业务代码零改动。

### 手语识别链（Transformer 架构 + 可选 ViT 视觉特征）

实际使用 **MediaPipe + ViT-B/16 + CSL Transformer Encoder** 混合方案：

1. `MediaPipeHandDetector` (`src/models/sign_language_model/mediapipe_detector.py`) — 21 关键点提取 + 检测框
2. `ViTFeatureExtractor` (`src/models/sign_language_model/vit_encoder.py`) — ViT-B/16 提取手部区域 768 维视觉特征（可选，`USE_VIT` 控制）
3. `CSLRecognizer` (`src/models/sign_language_model/csl_recognizer.py`) — Transformer Encoder 时序分类器（4层、8头自注意力、可学习位置编码）。有训练权重时用神经网络推理；权重缺失时自动降级为启发式规则。词汇表见 `DEFAULT_CSL_VOCABULARY`
4. `RealSignLanguageModel` (`src/models/sign_language_model/real_recognizer.py`) — 组合 MediaPipe + ViT + CSLTransformer，实现 `SignLanguageModel` 接口
4. `GestureRecognizer` (`src/models/sign_language_model/recognizer.py`) — 旧版 Hand-Gesture-19 (SigLIP2) 识别器，已不再主用但代码保留

### 文本翻译

- `Qwen2LoRAModel` (`src/models/text_model/qwen2_lora_model.py`) — Qwen2-0.5B-Instruct + LoRA 微调，词汇乱序→通顺中文句子。使用 chat template 推理
- `MockTranslateModel` (`src/models/text_model/mock_model.py`) — 基于映射表的轻量翻译器，零模型依赖。包含预定义 Token 映射表和句子模板
- `TranslateService` (`src/services/translate_service.py`) — 根据 `config.TRANSLATION_MODE`（qwen/mock/auto）决定使用哪个模型。默认为 mock（安全优先）

### 服务层

- `SignService` (`src/services/sign_service.py`) — 摄像头管理、帧缓冲、MediaPipe 检测集成、推理降级
- `TranslateService` (`src/services/translate_service.py`) — 懒加载翻译模型，类级别单例
- `SpeechService` (`src/services/speech_service.py`) — pyttsx3 离线 TTS，支持性别选择（Windows SAPI5）
- `VideoService` (`src/services/video_service.py`) — 视频下载 + 每 0.8s 抽帧
- `history_service.py` — SQLite CRUD（`data/mute.db`）
- `GenderService` / `EmotionService` — 占位服务，后者已弃用

### 双入口架构

- **WebSocket** (`src/websocket/handler.py`) — 摄像头实时推流，每连接一个 `StreamHandler` 实例，独立识别会话。客户端发送 base64 JPEG 帧，服务端 MediaPipe + CSL 逐帧分类
- **REST API** (`src/api/routes.py`) — 视频文件上传、翻译、TTS、历史记录 CRUD，前缀 `/api`

### 前端

`frontend/index.html` + `static/js/main.js` + `static/css/style.css` — 原生 HTML/CSS/JS，无框架。摄像头通过 `<canvas>` 渲染，WebSocket 双向通信，支持自动识别、手动识别、文本编辑、语音播放、历史管理。

## 关键配置

所有配置集中在 `src/config.py`，通过模块导入使用。关键项：

| 配置 | 说明 |
|------|------|
| `TRANSLATION_MODE` | `"auto"` / `"qwen"` / `"mock"`，默认 mock |
| `USE_VIT` | 是否启用 ViT-B/16 视觉特征提取（多模态融合） |
| `CAMERA_FPS` | 后端采集帧率 |
| `SIGN_AUTO_INTERVAL` | 自动识别间隔（秒），0 禁用 |
| `CSL_INPUT_DIM` | 模型输入维度：126=双手(左手63+右手63)，63=单手 |
| `CSL_VOCABULARY_PATH` | 训练词汇表路径，训练后自动生成 |
| `MODEL_CACHE_DIR` | HuggingFace 缓存目录（项目本地，非全局 ~/.cache） |

### 数据与训练管线

- `scripts/collect_data.py` — 交互式数据采集，摄像头录制手势关键点序列
- `scripts/train_csl.py` — CSLTransformer 训练脚本，类别权重+过采样+混淆矩阵
- `scripts/export_onnx.py` — ONNX FP32/int8 导出 + 推理延迟基准测试
- `data/gestures/` — 训练数据目录，`vocabulary.json` + `<手势名>/*.npy`
- 数据格式：126 维双手拼接（左手镜像后在前 63 维，右手在后 63 维），单手时缺失手填零

### CosyVoice2 TTS（可选）

- `src/services/cosyvoice_tts.py` — CosyVoice2 (Transformer+Flow Matching) 封装，pyttsx3 自动降级

### 部署

- `Dockerfile` + `docker-compose.yml` — 容器化部署
- `gradio_app.py` — Gradio 交互式 demo（视频上传+技术栈展示）

## SSL 证书

项目使用自签名证书 `key.pem` / `cert.pem`，通过 HTTPS 访问。浏览器需手动信任证书。
