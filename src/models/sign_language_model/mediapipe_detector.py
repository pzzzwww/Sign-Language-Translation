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

        # 【知识点：RunningMode】IMAGE=单帧独立检测，VIDEO=利用帧间连续性跟踪
        # 这里用 IMAGE 模式，适合 WebSocket 逐帧推流场景
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
            "MediaPipe Hands 检测器初始化完成 "
            "(min_det=%.2f, min_pres=%.2f, min_track=%.2f, max_hands=%d, model=%s)",
            min_detection_confidence,
            min_hand_presence_confidence,
            min_tracking_confidence,
            max_num_hands,
            model,
        )

    # ------------------------------------------------------------------
    # 检测
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> list[dict]:
        """对单帧图像进行手部检测（核心方法）。

        Args:
            image: (H, W, 3) BGR uint8 numpy array（OpenCV 默认颜色格式）

        Returns:
            hands_data: 检测到的手部列表，每项包含:
              - landmarks: (21, 3) 归一化坐标（xy 0-1, z 相对手腕深度）
              - bbox: (x, y, w, h) 像素坐标检测框
              - handedness: 'Left' / 'Right' 左右手标签
              - confidence: 检测置信度 0-1
              - landmarks_pixel: (21, 2) 实际像素坐标，用于前端绘制
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
        """
        重置跟踪状态。
        在 IMAGE 模式下无需重置，此方法为 API 兼容保留。
        （LIVE_STREAM 模式下需要调用此方法切换视频源）
        """
        # IMAGE 模式没有持久状态，无需操作
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

# =========================================================================
# 【知识点：特征工程 (Feature Engineering)】
# 以下两个函数将 MediaPipe 的 21 个关键点转换为机器学习模型可用的特征向量。
# 简单版：直接扁平化 21×3=63 维
# 增强版：额外提取手指角度、距离、伸展状态 → ~93 维
# =========================================================================

def extract_landmark_features(landmarks: np.ndarray) -> np.ndarray:
    """从 21 个关键点提取增强特征向量（用于手势分类）。

    特征组成（共 ~93 维）：
      - 原始坐标 63 维：21点 × 3 坐标(x,y,z)
      - 指尖到手腕距离 5 维：判断手指是否伸展
      - 手指弯曲角度 5 维：指尖-MCP-手腕夹角
      - 拇指角度 2 维：拇指方向用反正切表示
      - 手指展开度 10 维：5指之间两两距离
      - 掌心方向 3 维：法向量
      - 手指伸展状态 5 维：每个手指是否伸直

    【知识点：特征拼接 np.concatenate】将不同来源的特征向量拼成一个大向量
    """
    # 基础坐标特征
    coords = landmarks.flatten().astype(np.float32)

    # 【知识点：MediaPipe 关键点索引】
    # 手腕=0  拇指尖=4  食指尖=8  中指尖=12  无名指尖=16  小指尖=20
    fingertips = [4, 8, 12, 16, 20]
    finger_mcp = [2, 5, 9, 13, 17]   # 掌指关节（手指根部）

    # 指尖到手腕的欧几里得距离
    # 【知识点：np.linalg.norm】计算向量长度 = sqrt(x²+y²+z²)
    wrist = landmarks[0]
    tip_distances = np.array([
        np.linalg.norm(landmarks[tip] - wrist) for tip in fingertips
    ], dtype=np.float32)

    # 手指弯曲角度（用向量内积近似：指尖-MCP 与 MCP-手腕 的夹角）
    # 【知识点：np.dot 内积】正值=同向(伸展)，负值=反向(弯曲)
    finger_angles = np.array([
        np.dot(
            landmarks[tip] - landmarks[mcp],
            landmarks[mcp] - wrist,
        ) for tip, mcp in zip(fingertips, finger_mcp)
    ], dtype=np.float32)

    # 拇指方向（反正切表示角度）
    # 【知识点：arctan2】比 arctan 更稳，能正确处理四个象限
    thumb_vec = landmarks[4] - landmarks[2]
    thumb_angle_x = np.arctan2(thumb_vec[1], thumb_vec[0])
    thumb_angle_y = np.arctan2(thumb_vec[2], np.linalg.norm(thumb_vec[:2]) + 1e-8)

    # 指尖间两两距离（共 C(5,2)=10 对）
    tip_pairs = []
    for i in range(len(fingertips)):
        for j in range(i + 1, len(fingertips)):
            tip_pairs.append(
                np.linalg.norm(landmarks[fingertips[i]] - landmarks[fingertips[j]])
            )

    # 掌心朝向（用食指和小指 MCP 与手腕的叉积计算法向量）
    # 【知识点：叉积 np.cross】两个向量的叉积 = 它们所在平面的法向量
    palm_normal = np.cross(
        landmarks[5] - landmarks[0],   # 食指MCP - 手腕
        landmarks[17] - landmarks[0],  # 小指MCP - 手腕
    )
    if np.linalg.norm(palm_normal) > 1e-8:
        palm_normal = palm_normal / np.linalg.norm(palm_normal)  # 归一化

    # 手指伸展判断（PIP 关节 y 坐标小于 MCP = 手指伸直）
    # 注：这只在手掌朝下时准确，是个简化的近似判断
    pip_joints = [6, 10, 14, 18, 20]
    mcp_joints = [2, 5, 9, 13, 17]
    finger_extended = []
    for pip, mcp in zip(pip_joints, mcp_joints):
        extended = 1.0 if landmarks[pip][1] < landmarks[mcp][1] else 0.0
        finger_extended.append(extended)

    features = np.concatenate([
        coords,                          # 63
        tip_distances,                   # 5
        finger_angles,                   # 5
        [thumb_angle_x, thumb_angle_y],  # 2
        np.array(tip_pairs),             # 10
        palm_normal,                     # 3
        np.array(finger_extended),       # 5
    ])

    return features.astype(np.float32)


def landmarks_to_flatten(landmarks: np.ndarray) -> np.ndarray:
    """简单扁平化：(21, 3) 关键点 → (63,) 一维向量。直接展开，不提取额外特征。"""
    return landmarks.flatten().astype(np.float32)
