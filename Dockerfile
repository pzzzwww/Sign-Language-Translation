FROM python:3.10-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsndfile1 espeak \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 项目文件
COPY . .

# 创建数据目录
RUN mkdir -p audio tmp data onnx

# 暴露端口
EXPOSE 8000 7860

# 默认启动 FastAPI 后端
CMD ["python", "-m", "src.backend.main"]
