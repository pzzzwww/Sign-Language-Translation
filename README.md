# 基于Transformer的手语识别生成语音系统

> 实时手语识别与语音合成系统 — 让无声的表达被听见。

[![GitHub](https://img.shields.io/badge/GitHub-pzzzwww/Sign--Language--Translation-blue)](https://github.com/pzzzwww/Sign-Language-Translation)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **全球唯一**的开源端到端中文手语→语音实时翻译系统。学术界只做单一环节，商业方案闭源收费。本项目纯 CPU 运行，MIT 协议，识别→LLM 翻译→TTS 语音完整闭环。

## 项目简介

端到端的实时手语翻译与语音合成平台。用户通过**摄像头实时采集**或**上传视频文件**输入手语，系统自动识别手势词汇、翻译为自然中文语句，并合成为可播放的语音。

### 核心流程

```
摄像头/视频 → MediaPipe 手部关键点检测 → CSL Transformer 时序分类
    → Token 序列 → Qwen2-0.5B + LoRA 翻译 → 自然中文语句
    → pyttsx3 TTS → 语音播放
```

### 特性

- **实时手语识别**：摄像头帧推流（WebSocket），MediaPipe + CSL Transformer 逐帧分类
- **大模型翻译**：Qwen2-0.5B-Instruct + LoRA 微调，词汇序列重组为通顺中文句子
- **离线 TTS**：pyttsx3 本地语音合成，无需网络
- **自动降级**：模型权重缺失时自动切换启发式规则，系统不崩溃
- **双模式输入**：摄像头实时流 + 视频文件上传
- **翻译历史**：SQLite 持久化存储，支持回放、删除
- **预训练权重**：仓库内置 97.8% 验证准确率权重，克隆即可用

---

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI + Uvicorn | 异步 REST + WebSocket |
| 实时通信 | WebSocket | 摄像头帧推流，双向消息 |
| 手部检测 | MediaPipe | 21点手部关键点实时提取（CPU 30fps+） |
| 视觉编码 | ViT-B/16 | 768维手部区域视觉特征（可选，默认关闭） |
| 时序分类 | CSL Transformer Encoder | 6层 8头自注意力 + 可学习位置编码，~284万参数 |
| 文本翻译 | Qwen2-0.5B-Instruct + LoRA | PEFT 微调（rank=8），adapter ~2MB |
| 语音合成 | pyttsx3 | Windows SAPI5 离线 TTS，支持性别选择 |
| 数据存储 | SQLite | 翻译历史 CRUD |
| 前端 | 原生 HTML/CSS/JS | 零框架依赖 |

### 设计模式

- **Strategy 模式**：`SignLanguageModel` / `TextTranslateModel` 抽象接口，工厂函数创建实例
- **Facade 模式**：`RealSignLanguageModel` 组合 MediaPipe + ViT + CSL 子系统
- **降级容错**：每层模型不可用时自动切换备选方案

---

## 快速启动

### 环境要求

- Python 3.10+
- PyTorch 2.5+
- 摄像头（实时模式需要）

### 安装与运行

```bash
# 克隆
git clone https://github.com/pzzzwww/Sign-Language-Translation.git
cd Sign-Language-Translation

# 安装依赖
pip install -r requirements.txt

# 启动服务（预训练权重已在仓库中）
python -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8008
```

浏览器访问 **http://localhost:8008**，点击"开始采集"，摄像头前比划手势即可。

> Qwen2-0.5B 首次启动自动下载（约 1GB）。如不想下载，修改 `src/config.py` 中 `TRANSLATION_MODE = "mock"`。

---

## 模型训练

### 采集数据

```bash
python scripts/collect_data.py
```

每个手势录 20-50 段，采集时故意变换角度、距离、速度以提升泛化能力。

### 训练

```bash
python scripts/train_csl.py
```

训练参数（`--epochs`、`--batch`、`--lr`）可在命令行覆盖：

```bash
python scripts/train_csl.py --epochs 60 --batch 16 --lr 5e-4
```

训练自动完成：滑动窗口切分 → 数据增强 → 类别加权/过采样 → 训练 → 早停 → 混淆矩阵。

当前预训练权重：**12 类手势，验证准确率 97.78%**。

### 团队协作流程

```
成员采集数据 → 发给你 → 你合并训练 → 推送新权重 → 成员 git pull
```

1. 成员运行 `python scripts/collect_data.py` 采集手势
2. 成员将 `data/gestures/` 打包发给你
3. 你合并所有数据到 `data/gestures/`（保持 `vocabulary.json` 一致）
4. 运行 `python scripts/train_csl.py` 重新训练
5. `git push` 更新仓库权重

成员只负责采集，不需要训练。

---

## 项目结构

```
├── src/
│   ├── backend/main.py           # FastAPI 应用入口
│   ├── api/routes.py             # REST API 路由
│   ├── websocket/handler.py      # WebSocket 连接管理 + 实时识别
│   ├── services/                 # 业务服务层
│   │   ├── sign_service.py       # 手语识别服务
│   │   ├── translate_service.py  # Token → 中文翻译
│   │   ├── speech_service.py     # TTS 语音合成
│   │   └── history_service.py    # 翻译历史 CRUD
│   ├── models/
│   │   ├── sign_language_model/  # 手语识别模型
│   │   │   ├── csl_recognizer.py       # CSL Transformer 时序分类
│   │   │   ├── mediapipe_detector.py   # MediaPipe 手部关键点
│   │   │   ├── vit_encoder.py          # ViT-B/16 视觉编码
│   │   │   └── real_recognizer.py      # 三合一识别器
│   │   └── text_model/           # 文本翻译模型
│   │       ├── qwen2_lora_model.py     # Qwen2 + LoRA
│   │       ├── mock_model.py           # 轻量降级翻译
│   │       └── lora_word2sent/         # LoRA 适配器权重
│   ├── interfaces/               # 抽象接口
│   └── config.py                 # 集中配置
├── scripts/
│   ├── collect_data.py           # 数据采集
│   ├── train_csl.py              # 模型训练
│   ├── export_onnx.py            # ONNX 导出
│   └── gradio_app.py             # Gradio Demo
├── frontend/                     # 原生 HTML/CSS/JS
├── requirements.txt
└── README.md
```

---

## API 概览

### REST API（`/api`）

| 方法 | 端点 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 健康检查 |
| `GET` | `/api/status` | 模型加载状态 |
| `POST` | `/api/translate` | 词汇列表 → 翻译句子 |
| `POST` | `/api/tts` | 文本 → WAV 音频 |
| `POST` | `/api/process-video` | 上传视频 → 识别翻译 |
| `GET` | `/api/history` | 翻译历史列表 |
| `DELETE` | `/api/history/{id}` | 删除历史记录 |

### WebSocket（`/ws/stream`）

| action | 说明 |
|--------|------|
| `start_capture` | 启动摄像头推流 |
| `stop` | 停止摄像头 |
| `recognize` | 执行识别 + 翻译 |
| `process_frame` | 发送单帧（base64 JPEG） |
| `confirm_translate` | 确认翻译文本 |
| `generate_audio` | 生成语音 |

---

## 配置说明

所有配置集中 `src/config.py`：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `TRANSLATION_MODE` | `"qwen"` | 翻译模式：qwen / mock |
| `USE_VIT` | `False` | 是否启用 ViT 视觉特征 |
| `CAMERA_FPS` | `12` | 采集帧率 |
| `CSL_INPUT_DIM` | `126` | 输入维度（双手 126 / 单手 63） |
| `CSL_CONFIDENCE_THRESHOLD` | `0.55` | 置信度阈值 |
| `CSL_STABILITY_THRESHOLD` | `5` | 连续 N 帧一致才确认 |

---

## 常见问题

**Q: 摄像头打不开？**
A: 修改 `src/config.py` 中 `CAMERA_INDEX`，Windows 可能为 0 或 1。

**Q: 模型加载很慢？**
A: 首次启动 Qwen2 需下载约 1GB。设为 `TRANSLATION_MODE = "mock"` 可跳过。

**Q: 手语识别不准？**
A: 采集更多数据重新训练（每类至少 50 个样本，变换角度、距离、速度）。

**Q: 语音合成没有声音？**
A: pyttsx3 依赖系统 TTS 引擎。Windows 使用 SAPI5，Linux 需 `apt install espeak`。

---

## License

[MIT](LICENSE)
