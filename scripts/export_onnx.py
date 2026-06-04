"""
CSLTransformer ONNX 导出 + int8 量化。

导出训练好的 CSL Transformer 模型到 ONNX 格式，支持:
  - FP32 ONNX（标准推理）
  - int8 动态量化 ONNX（推理加速 2-4x，模型体积减半）

Usage:
    # 导出 FP32
    python scripts/export_onnx.py --mode fp32

    # 导出 int8 量化
    python scripts/export_onnx.py --mode int8

    # 验证导出结果
    python scripts/export_onnx.py --verify
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def export_fp32(output_dir: str = "onnx") -> None:
    """导出 FP32 ONNX 模型。"""
    import torch
    from src.models.sign_language_model.csl_recognizer import CSLTransformer

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    logger.info("创建 CSLTransformer 模型...")
    model = CSLTransformer(num_classes=100, input_dim=63)
    model.eval()

    dummy = torch.randn(1, 30, 63)
    onnx_path = out / "csl_transformer_fp32.onnx"

    logger.info("导出 ONNX (FP32): %s", onnx_path)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["landmark_sequence"],
        output_names=["logits"],
        dynamic_axes={
            "landmark_sequence": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )
    logger.info("FP32 ONNX 导出完成 (%.1f KB)", onnx_path.stat().st_size / 1024)
    verify_onnx(onnx_path)


def export_int8(output_dir: str = "onnx") -> None:
    """导出 int8 动态量化 ONNX 模型。"""
    import torch
    from src.models.sign_language_model.csl_recognizer import CSLTransformer

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    # 1. 先导出 FP32 ONNX
    logger.info("第一步: 导出 FP32 ONNX...")
    model = CSLTransformer(num_classes=100, input_dim=63)
    model.eval()

    dummy = torch.randn(1, 30, 63)
    fp32_path = out / "csl_transformer_fp32.onnx"

    torch.onnx.export(
        model, dummy, str(fp32_path),
        input_names=["landmark_sequence"],
        output_names=["logits"],
        dynamic_axes={
            "landmark_sequence": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )

    # 2. ONNX Runtime int8 动态量化
    logger.info("第二步: int8 动态量化...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        logger.error("onnxruntime 未安装，请运行: pip install onnxruntime")
        return

    int8_path = out / "csl_transformer_int8.onnx"
    try:
        quantize_dynamic(
            str(fp32_path),
            str(int8_path),
            weight_type=QuantType.QInt8,
        )
        logger.info("int8 ONNX 导出完成 (%.1f KB)", int8_path.stat().st_size / 1024)
        verify_onnx(int8_path)

        fp32_size = fp32_path.stat().st_size / 1024
        int8_size = int8_path.stat().st_size / 1024
        logger.info(
            "量化对比: FP32=%.0fKB → int8=%.0fKB (压缩率 %.1fx)",
            fp32_size, int8_size, fp32_size / int8_size,
        )
    except Exception as e:
        logger.warning("int8 量化失败 (onnx/onnxruntime 版本兼容): %s", e)
        logger.info("FP32 ONNX 仍可使用: %s", fp32_path)
        logger.info("提示: pip install --upgrade onnx onnxruntime")


def verify_onnx(onnx_path: str | Path) -> None:
    """验证 ONNX 模型的输入输出形状。"""
    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        inputs = [i.name for i in model.graph.input]
        outputs = [o.name for o in model.graph.output]
        logger.info("ONNX 验证通过: inputs=%s, outputs=%s", inputs, outputs)
    except ImportError:
        logger.warning("onnx 未安装，跳过验证。pip install onnx")


def benchmark_onnx(onnx_path: str | Path, n_runs: int = 100) -> None:
    """基准测试 ONNX vs PyTorch 推理延迟。"""
    import time
    import torch
    import onnxruntime as ort

    from src.models.sign_language_model.csl_recognizer import CSLTransformer

    onnx_path = Path(onnx_path)

    # PyTorch 基准
    model = CSLTransformer(num_classes=100, input_dim=63)
    model.eval()
    x = torch.randn(1, 30, 63)

    with torch.no_grad():
        # warmup
        for _ in range(10):
            model(x)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(x)
        torch_time = (time.perf_counter() - t0) / n_runs * 1000

    # ONNX 基准
    session = ort.InferenceSession(str(onnx_path))
    ort_inputs = {"landmark_sequence": x.numpy()}

    for _ in range(10):
        session.run(None, ort_inputs)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, ort_inputs)
    onnx_time = (time.perf_counter() - t0) / n_runs * 1000

    logger.info("推理延迟对比 (%d runs):", n_runs)
    logger.info("  PyTorch: %.2f ms", torch_time)
    logger.info("  ONNX:    %.2f ms (加速 %.1fx)", onnx_time, torch_time / onnx_time)


def main() -> None:
    parser = argparse.ArgumentParser(description="CSLTransformer ONNX 导出")
    parser.add_argument(
        "--mode", choices=["fp32", "int8"], default="fp32",
        help="导出模式 (default: fp32)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="验证已导出的 ONNX 文件",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="基准测试 ONNX vs PyTorch 推理延迟",
    )
    parser.add_argument(
        "--output", default="onnx",
        help="输出目录 (default: onnx)",
    )
    args = parser.parse_args()

    if args.benchmark:
        int8_path = Path(args.output) / "csl_transformer_int8.onnx"
        if int8_path.exists():
            benchmark_onnx(int8_path)
        else:
            fp32_path = Path(args.output) / "csl_transformer_fp32.onnx"
            if fp32_path.exists():
                benchmark_onnx(fp32_path)
            else:
                logger.error("ONNX 文件不存在，请先导出")
    elif args.verify:
        for p in Path(args.output).glob("*.onnx"):
            verify_onnx(p)
    elif args.mode == "int8":
        export_int8(args.output)
    else:
        export_fp32(args.output)


if __name__ == "__main__":
    main()
