"""
【知识点：MediaPipe 手部关键点检测】

Google 的轻量级 ML 框架，能在 CPU 上实时检测手部 21 个关键点。
这里用的是 task-based API（mediapipe 0.10.x 新接口）。

核心概念：
  - 21 个关键点：手腕(0) + 拇指4关节(1-4) + 食指4关节(5-8)
    + 中指4关节(9-12) + 无名指4关节(13-16) + 小指4关节(17-20)
  - 归一化坐标：x,y 在 0-1 之间（相对图像宽高），z 相对于手腕深度
  - 置信度：0-1，表示模型对该检测结果的把握

输出数据用于：
  1. 前端 Canvas 叠加层绘制检测框
  2. CSL 手语识别模型输入（关键点序列）
  3. 实时字幕 Token 显示
"""

from __future__ import annotations

import logging
import os
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# 【知识点：optional import / 懒加载】运行时检查依赖是否安装，
# 未安装时给出清晰错误提示，而不是直接崩溃
try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        HandLandmarkerResult,
        RunningMode,
    )
    from mediapipe import Image as MPImage
    from mediapipe import ImageFormat
    _MP_AVAILABLE = True
except ImportError as e:
    _MP_AVAILABLE = False
    mp = None
    HandLandmarker = None
    HandLandmarkerOptions = None
    HandLandmarkerResult = None
    RunningMode = None
    BaseOptions = None
    MPImage = None
    ImageFormat = None

# ---- 模型路径（环境变量 > 默认位置）----
# 【知识点：.task 格式】MediaPipe 的 TFLite 模型文件，包含手部关键点检测模型权重
_MODEL_FILENAME = "hand_landmarker.task"
_DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pretrained"
)
# 【知识点：环境变量配置】可通过 MEDIAPIPE_HAND_MODEL 环境变量指定模型位置
_MODEL_PATH = os.environ.get(
    "MEDIAPIPE_HAND_MODEL",
    os.path.join(_DEFAULT_MODEL_DIR, _MODEL_FILENAME),
)


class MediaPipeHandDetector:
    """
    MediaPipe Hands 检测器封装（task-based API）。

    每帧检测手部，返回：
      - 21个关键点 (x, y, z) 归一化坐标
      - 手部边界框 (x, y, w, h) 像素坐标
      - 左右手标签 + 置信度

    Usage:
        detector = MediaPipeHandDetector()
        hands = detector.detect(bgr_image)
        for hand in hands:
            print(hand['landmarks'].shape)  # (21, 3)
            print(hand['bbox'])             # (x, y, w, h)
            print(hand['handedness'])       # 'Left' / 'Right'
            print(hand['confidence'])       # 0-1
        detector.close()
    """

    def __init__(
        self,
        model_path: str | None = None,
        min_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 2,
    ) -> None:
        """初始化 MediaPipe 手部检测器。

        参数说明：
          model_path: .task 模型文件路径，None 则用默认
          min_detection_confidence: 手部检测置信度阈值（0-1），低于此值的手不被检测
          min_hand_presence_confidence: 手部存在置信度阈值
          min_tracking_confidence: 手部跟踪置信度阈值（VIDEO模式下用，IMAGE模式忽略）
          max_num_hands: 最多检测的手数量，默认2（同时支持双手）
        """
        if not _MP_AVAILABLE:
            raise ImportError(
                "mediapipe 未安装，请运行: pip install mediapipe>=0.10.7"
            )

        model = model_path or _MODEL_PATH
        if not os.path.isfile(model):
            raise FileNotFoundError(
                f"MediaPipe 手部模型未找到: {model}\n"
                "请下载 hand_landmarker.task 到 pretrained/ 目录，"
                "或设置环境变量 MEDIAPIPE_HAND_MODEL"
            )

        # 【知识点：RunningMode】VIDEO=利用帧间连续性跟踪，检测更稳定
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model),
            running_mode=RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        logger.info(
            "MediaPipe Hands 检测器初始化完成 (IMAGE 模式)"
            "(min_det=%.2f, min_pres=%.2f, max_hands=%d, model=%s)",
            min_detection_confidence,
            min_hand_presence_confidence,
            max_num_hands,
            model,
        )

    # ------------------------------------------------------------------
    # 检测
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> list[dict]:
        """
        一张图片 → detect() → 每只手的 4 样数据：
                         ① 关键点坐标    → 拼 126 维 → 模型分类
                         ② 像素坐标      → 前端画点
                         ③ 检测框        → 前端画框 + ViT 裁切手部区域
                         ④ 左右手+置信度  → 左手镜像 + 过滤低质量检测 + 框颜色
        """
        # 【知识点：BGR→RGB】OpenCV 默认 BGR，MediaPipe 需要 RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 【知识点：MediaPipe Image 封装】将 numpy 数组包装为 MediaPipe 可识别的图像对象
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)
        result: HandLandmarkerResult = self._landmarker.detect(mp_image)

        hands_data: list[dict] = []

        if not result.hand_landmarks:
            return hands_data

        h, w = image.shape[:2]

        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            # ---- 1. 提取关键点归一化坐标 (21, 3) ----
            # 【知识点：MediaPipe 关键点】每个点有 x,y,z，x/width, y/height, z 相对手腕
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                dtype=np.float32,
            )

            # ---- 2. 转换为实际像素坐标（用于前端绘制）----
            landmarks_pixel = landmarks[:, :2].copy()
            landmarks_pixel[:, 0] *= w  # 归一化 x → 像素 x
            landmarks_pixel[:, 1] *= h  # 归一化 y → 像素 y

            # ---- 3. 计算边界框（所有关键点围起来的矩形）----
            # 【知识点：Bounding Box】用最小外接矩形框住手部区域
            x_min, y_min = landmarks_pixel.min(axis=0)
            x_max, y_max = landmarks_pixel.max(axis=0)

            # 向外扩展 10%，避免裁剪太紧
            margin_x = (x_max - x_min) * 0.1
            margin_y = (y_max - y_min) * 0.1
            bx = max(0, int(x_min - margin_x))
            by = max(0, int(y_min - margin_y))
            bw = min(w - bx, int(x_max - x_min + 2 * margin_x))
            bh = min(h - by, int(y_max - y_min + 2 * margin_y))

            # ---- 4. 确定左右手 ----
            # 【知识点：handedness】MediaPipe 可自动区分左右手
            handedness_list = result.handedness[idx]
            if handedness_list:
                hand_label = handedness_list[0].category_name  # 'Left' / 'Right'
                hand_conf = handedness_list[0].score
            else:
                hand_label = "Unknown"
                hand_conf = 0.0

            hands_data.append({
                "landmarks": landmarks,
                "landmarks_pixel": landmarks_pixel,
                "bbox": (bx, by, bw, bh),
                "handedness": hand_label,
                "confidence": float(hand_conf),
                "index": idx,
            })

        return hands_data

    # ------------------------------------------------------------------
    # 辅助：绘制检测结果到图像上
    # ------------------------------------------------------------------

    def draw_detection(
        self,
        image: np.ndarray,
        hands_data: list[dict],
        token_labels: list[str] | None = None,
    ) -> np.ndarray:
        """
        将手部检测框和关键点绘制到图像上（用于调试/录制）。

        颜色方案：
          - 正常识别 (confidence >= 0.7): 绿色
          - 置信度低 (0.4 <= conf < 0.7): 黄色
          - 未识别 (conf < 0.4 或无 token): 红色

        Args:
            image: BGR 图像
            hands_data: detect() 返回的手部数据
            token_labels: 每个手部对应的 Token 标签列表

        Returns:
            绘制了检测信息的 BGR 图像
        """
        result = image.copy()
        labels = token_labels or [""] * len(hands_data)

        for i, hand in enumerate(hands_data):
            bx, by, bw, bh = hand["bbox"]
            confidence = hand["confidence"]
            token = labels[i] if i < len(labels) else ""

            # 颜色
            if token and confidence >= 0.7:
                color = (0, 255, 0)  # 绿色 BGR
            elif token and confidence >= 0.4:
                color = (0, 215, 255)  # 黄色 BGR
            else:
                color = (0, 0, 255)  # 红色 BGR

            # 绘制边界框
            cv2.rectangle(result, (bx, by), (bx + bw, by + bh), color, 2)

            # 绘制标签
            label_text = token if token else f"{hand['handedness']}"
            label_text += f" {confidence:.2f}"
            cv2.putText(
                result,
                label_text,
                (bx, by - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            # 绘制关键点
            for px, py in hand["landmarks_pixel"]:
                cv2.circle(
                    result, (int(px), int(py)), 3, color, -1,
                )

        return result

    # ------------------------------------------------------------------
    # 资源管理
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """重置跟踪状态（IMAGE 模式无需操作，API 兼容保留）。"""
        pass

    def close(self) -> None:
        """释放 MediaPipe 资源。"""
        if self._landmarker is not None:
            self._landmarker.close()
        logger.info("MediaPipe Hands 检测器已释放")

    def __enter__(self) -> "MediaPipeHandDetector":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ------------------------------------------------------------------
# 关键点特征提取工具
# ------------------------------------------------------------------

def landmarks_to_flatten(landmarks: np.ndarray) -> np.ndarray:
    """简单扁平化：(21, 3) 关键点 → (63,) 一维向量。直接展开，不提取额外特征。"""
    return landmarks.flatten().astype(np.float32)
