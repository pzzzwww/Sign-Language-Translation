"""
Gradio 演示页面 — 手语识别 + 翻译 + 语音合成一站式体验。

启动方式:
    python gradio_app.py
    或: python gradio_app.py --share  (生成公网可分享链接)

技术栈展示:
    - MediaPipe 手部关键点检测
    - ViT-B/16 视觉特征提取 (可选)
    - Transformer Encoder 时序手势分类
    - Qwen2-1.5B + LoRA 文本翻译
    - pyttsx3 / CosyVoice2 语音合成
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pipeline():
    """懒加载手语识别流水线。"""
    from src.services.sign_service import SignService
    from src.services.translate_service import TranslateService
    from src.services.speech_service import SpeechService
    from src.models import get_sign_language_model

    sign_model = get_sign_language_model()
    sign_model.load()
    sign_service = SignService(sign_model)
    translate_service = TranslateService()
    speech_service = SpeechService()

    return sign_service, translate_service, speech_service


_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = load_pipeline()
    return _pipeline


def process_video(video_path: str) -> tuple[str, str, str | None]:
    """处理上传的手语视频。"""
    if not video_path:
        return "", "请上传视频文件", None

    sign_svc, translate_svc, speech_svc = get_pipeline()

    # 抽帧
    from src.services.video_service import VideoService
    video_svc = VideoService()
    frame_paths = video_svc.extract_frames(video_path)
    video_svc.clear_frames()

    if not frame_paths:
        return "", "未能从视频中提取到有效帧", None

    # 手语识别
    tokens = sign_svc.process_frame_files(frame_paths)
    if not tokens:
        return "无识别结果", "未检测到手语 Token", None

    tokens_str = " | ".join(tokens)

    # 翻译
    try:
        sentence = translate_svc.translate(tokens)
    except Exception as e:
        return tokens_str, f"翻译失败: {e}", None

    # TTS
    try:
        audio_path = "tmp/gradio_output.wav"
        Path(audio_path).parent.mkdir(exist_ok=True)
        speech_svc.synthesize_to_file(sentence, audio_path)
    except Exception:
        audio_path = None

    return tokens_str, sentence, audio_path


def recognize_from_webcam(image: np.ndarray) -> tuple[np.ndarray, str, str]:
    """处理摄像头帧。"""
    if image is None:
        return image, "", ""

    sign_svc, translate_svc, _ = get_pipeline()

    # MediaPipe 检测
    model = sign_svc._model
    if not hasattr(model, "predict_frame"):
        return image, "模型未就绪", ""

    result = model.predict_frame(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # 绘制检测框
    if result["hands_data"] and model.detector:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_bgr = model.detector.draw_detection(
            image_bgr,
            result["hands_data"],
            [h.get("token") for h in result["hands_data"]],
        )
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    all_tokens = result["all_tokens"]
    tokens_str = " | ".join(all_tokens)

    # 翻译 (有 Token 时)
    sentence = ""
    if len(all_tokens) >= 2:
        try:
            sentence = translate_svc.translate(all_tokens)
        except Exception:
            sentence = ""

    return image, tokens_str, sentence


def create_demo():
    """构建 Gradio 界面。"""
    with gr.Blocks(
        title="基于Transformer的手语识别生成语音系统",
        theme=gr.themes.Soft(),
        css="""
        .token-box { font-size: 20px; color: #2196F3; }
        .main-title { text-align: center; }
        """,
    ) as demo:
        gr.Markdown(
            """
            # 🤟 基于 Transformer 的手语识别生成语音系统
            ### Vision Transformer + Transformer Encoder + Qwen2 + TTS
            """
        )

        with gr.Tabs():
            # ---- Tab 1: 视频上传 ----
            with gr.TabItem("📹 视频上传"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="上传手语视频")
                        btn_process = gr.Button("🚀 开始识别", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("### 🔤 识别 Token")
                        token_output = gr.Textbox(
                            label="手语 Token", lines=3, elem_classes="token-box"
                        )
                        gr.Markdown("### 💬 翻译结果")
                        sentence_output = gr.Textbox(
                            label="翻译句子", lines=2
                        )
                        audio_output = gr.Audio(label="🔊 语音合成", type="filepath")

                btn_process.click(
                    process_video,
                    inputs=[video_input],
                    outputs=[token_output, sentence_output, audio_output],
                )

            # ---- Tab 2: 技术栈 ----
            with gr.TabItem("📋 技术栈"):
                gr.Markdown(
                    """
                    ## 技术架构

                    | 环节 | 技术 | 说明 |
                    |------|------|------|
                    | 手部检测 | **MediaPipe Hands** | 21 关键点 + 检测框 |
                    | 视觉编码 | **ViT-B/16** (可选) | 手部区域 768 维特征 |
                    | 时序分类 | **Transformer Encoder** | 4层/8头自注意力 + 可学习位置编码 |
                    | 文本翻译 | **Qwen2-1.5B + LoRA** | 词汇乱序→自然中文 |
                    | 语音合成 | **CosyVoice2 / pyttsx3** | Transformer + Flow Matching / SAPI5 |
                    | 推理加速 | **ONNX Runtime + int8 量化** | 推理加速 2-4x |
                    | 部署 | **FastAPI + Docker** | 容器化部署 |

                    ## 项目特点

                    - ✅ **全链路 Transformer 架构**：视觉 → 时序 → 文本 → 语音
                    - ✅ **多模态融合**：ViT 视觉特征 + MediaPipe 关键点
                    - ✅ **消融实验支持**：可独立开关 ViT 模块对比效果
                    - ✅ **优雅降级**：模型权重缺失时自动切换启发式规则
                    - ✅ **ONNX 部署**：支持 int8 量化导出，CPU 可运行

                    ## 启动方式

                    ```bash
                    # FastAPI 后端
                    python -m src.backend.main

                    # Gradio 演示
                    python gradio_app.py --share

                    # ONNX 导出
                    python scripts/export_onnx.py --mode int8
                    ```
                    """
                )

        return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="手语识别 Gradio 演示")
    parser.add_argument("--share", action="store_true", help="生成公网分享链接")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
