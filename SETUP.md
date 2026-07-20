# 项目部署指南

从零开始：拿到源码 → 装环境 → 启动 → 浏览器访问。

---

## 一、环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.10+ |
| PyTorch | 2.5+（有显卡装 CUDA 版，没有装 CPU 版） |
| 摄像头 | 实时模式需要（视频上传模式不需要） |
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

### 5. Linux 额外步骤

pyttsx3 语音合成需要 espeak：

```bash
# Ubuntu / Debian
sudo apt install espeak -y

# CentOS / RHEL
sudo yum install espeak -y
```

Windows 和 macOS 不需要，系统自带语音引擎。

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

### 不想走 HTTPS？

```bash
# Linux / macOS
NO_SSL=1 python -m src.backend.main

# Windows PowerShell
$env:NO_SSL=1; python -m src.backend.main
```

### 国际用户下载慢？

项目默认用 `hf-mirror.com` 国内镜像。国际用户切回官方源：

```bash
# Linux / macOS
HF_ENDPOINT=https://huggingface.co python -m src.backend.main

# Windows PowerShell
$env:HF_ENDPOINT="https://huggingface.co"; python -m src.backend.main
```

---

## 四、访问使用

1. 浏览器打开 **https://localhost:8000**
2. 提示证书不安全 → 点「高级 → 继续访问」（自签名 SSL 证书，正常现象）
3. 点「开始采集」，允许摄像头权限
4. 对着摄像头比划手势，系统自动识别并显示翻译结果
5. 点「确认文本」→「生成语音」可播放语音

---

## 五、常见问题

**Q: `ImportError: No module named 'torch'`**
A: PyTorch 没装或装错版本。按步骤 3 重新安装，注意 CUDA 版本要和显卡驱动匹配。

**Q: 启动报错 `cannot import name ...`**
A: 依赖版本不对。重新 `pip install -r requirements.txt`，不要手动装包。

**Q: 摄像头打不开**
A: 修改 `src/config.py` 中 `CAMERA_INDEX`，Windows 通常为 0 或 1。

**Q: 模型下载很慢 / 超时**
A: 项目已默认走 `hf-mirror.com` 镜像。如果还是慢，检查网络或用 `TRANSLATION_MODE = "mock"` 跳过。

**Q: 语音合成没有声音**
A: Linux 需 `sudo apt install espeak`。Windows/macOS 系统自带，检查系统音量和静音设置。

**Q: 浏览器打不开页面**
A: 检查防火墙是否放行 8000 端口。云服务器还需要在安全组入方向放行 8000（TCP）。

**Q: 内存不够 / OOM**
A: Qwen2-0.5B 约占 1G 内存。4G 服务器够用，2G 以下建议用 `TRANSLATION_MODE = "mock"`。

---

## 六、手势词汇表

当前支持 26 个手势：

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| 我 | 你 | 喜欢 | 谢谢 | 对不起 | 没关系 | 你好 | 为什么 | 谁 |
| 在 | 去 | 吃 | 很 | 会 | 大家 | 一起 | 面包 | 上次 |
| 开心 | 祝 | 帮助 | 请 | 问 | 快点 | 想要 | 手语 | |

可通过 `scripts/collect_data.py` 采集新手势，`scripts/train_csl.py` 重新训练扩展词汇表。
