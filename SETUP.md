# 项目部署指南

从零开始：拿到源码 → 装环境 → 启动 → 浏览器访问。

---

## 一、环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.10+ |
| PyTorch | 2.5+（有显卡装 CUDA 版，没有装 CPU 版） |
| 摄像头 | 实时模式需要 |
| 内存 | 4G+（Qwen2-0.5B 约占 1G） |
| 显卡 | 可选，有 GPU 推理更快，没有自动走 CPU |

---

## 二、安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/pzzzwww/Sign-Language-Translation.git
cd Sign-Language-Translation
```

### 2. 创建虚拟环境

**方式 A：conda（推荐）**

```bash
conda create -n transformer python=3.10 -y
conda activate transformer
```

**方式 B：venv**

```bash
python3.10 -m venv .venv
source .venv/bin/activate    # Linux/macOS
.venv\Scripts\activate       # Windows
```

### 3. 安装 PyTorch

有 NVIDIA 显卡（根据 CUDA 版本选择，查 [pytorch.org](https://pytorch.org/get-started/locally/)）：

```bash
# CUDA 12.1 示例
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

无显卡（CPU 版）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. 安装项目依赖

```bash
pip install -r requirements.txt
```

---

## 三、启动服务

```bash
python -m src.backend.main
```

首次启动会自动下载 Qwen2-0.5B 模型（约 1GB），默认走 `hf-mirror.com` 国内镜像加速。下载完成后服务启动。

看到以下输出说明成功：

```
============================================================
  基于Transformer的手语识别生成语音系统 v0.3
  访问地址: https://localhost:8000
============================================================

INFO: Application startup complete.
INFO: Uvicorn running on https://0.0.0.0:8000
```

### 不想下载 1GB 模型？

修改 `src/config.py`：

```python
TRANSLATION_MODE = "mock"   # 用映射表代替，零模型依赖
```

### 国际用户下载慢？

项目默认用 `hf-mirror.com` 国内镜像。国际用户切回官方源：

```bash
# Windows PowerShell
$env:HF_ENDPOINT="https://huggingface.co"; python -m src.backend.main
```



---

## 四、手势词汇表

当前支持 26 个手势：

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| 我 | 你 | 喜欢 | 谢谢 | 对不起 | 没关系 | 你好 | 为什么 | 谁 |
| 在 | 去 | 吃 | 很 | 会 | 大家 | 一起 | 面包 | 上次 |
| 开心 | 祝 | 帮助 | 请 | 问 | 快点 | 想要 | 手语 | |

可通过 `scripts/collect_data.py` 采集新手势，`scripts/train_csl.py` 重新训练扩展词汇表。
