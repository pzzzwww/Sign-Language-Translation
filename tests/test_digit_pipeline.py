"""
数字手势识别管线验证脚本
========================
验证 MediaPipe 检测 -> 启发式数字分类 -> Token 输出
"""
import logging
import sys
import io
import numpy as np
import cv2

# 强制 UTF-8 输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 开启详细日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

from models.sign_language_model.mediapipe_detector import (
    MediaPipeHandDetector, landmarks_to_flatten,
)
from models.sign_language_model.csl_recognizer import CSLRecognizer
from models.sign_language_model.real_recognizer import RealSignLanguageModel

print("\n" + "=" * 60)
print("  数字手势识别管线验证")
print("=" * 60)

# ==============================================
# 1. CSL 识别器加载诊断
# ==============================================
print("\n[Step 1] CSL 识别器加载诊断")
print("-" * 40)
recognizer = CSLRecognizer(
    model_path="models/sign_language_model/pretrained/csl_model.pt",
    confidence_threshold=0.4,
    stability_threshold=2,   # 降低门槛便于测试
    cooldown_frames=5,        # 降低冷却便于测试
)
recognizer.load()
diag = recognizer.get_diagnostic_info()
print(f"  识别模式:       {diag['model_type']}")
print(f"  已训练权重:     {diag['has_trained_weights']}")
print(f"  词汇表大小:     {diag['vocabulary_size']}")
print(f"  前10个词汇:     {diag['vocabulary_first_10']}")
print(f"  最小帧数:       {diag['min_frames_required']}")

# 验证: 词汇表必须包含 "1"-"5"
assert "1" in recognizer.vocabulary, "FAIL: 词汇表缺少'1'"
assert "5" in recognizer.vocabulary, "FAIL: 词汇表缺少'5'"
print("  [OK] 数字词汇 1-5 已加入词汇表")

# 验证: 启发式模式必须激活
assert not diag['has_trained_weights'], "FAIL: 应处于启发式模式"
print("  [OK] 启发式模式已激活 (角度法数字分类)")

# ==============================================
# 2. MediaPipe 手部检测器加载诊断
# ==============================================
print("\n[Step 2] MediaPipe 手部检测器加载诊断")
print("-" * 40)
detector = MediaPipeHandDetector(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    max_num_hands=1,
)
print("  [OK] MediaPipe 检测器加载成功")
print(f"  模型文件: hand_landmarker.task")

# ==============================================
# 3. 用合成关键点验证启发式数字分类
# ==============================================
print("\n[Step 3] 启发式数字分类验证 (合成关键点)")
print("-" * 40)

FINGER_TIP = [4, 8, 12, 16, 20]
FINGER_PIP = [3, 6, 10, 14, 18]
FINGER_MCP = [2, 5, 9, 13, 17]

def make_landmarks(extended_fingers):
    """
    生成模拟关键点 (21, 3)。
    extended_fingers: [thumb, index, middle, ring, pinky] bool 列表
    伸展的手指：指尖远离手腕方向，形成大角度
    弯曲的手指：指尖靠近MCP
    """
    lm = np.zeros((21, 3), dtype=np.float32)
    # 手腕在底部
    lm[0] = [0.5, 0.85, 0.0]
    # MCP 关节
    lm[2]  = [0.32, 0.72, 0.0]   # thumb MCP
    lm[5]  = [0.42, 0.67, 0.0]   # index MCP
    lm[9]  = [0.50, 0.65, 0.0]   # middle MCP
    lm[13] = [0.58, 0.67, 0.0]   # ring MCP
    lm[17] = [0.68, 0.72, 0.0]   # pinky MCP

    # 中间连接点
    lm[1]  = (lm[0] + lm[2]) / 2
    lm[6]  = (lm[5] + lm[9]) / 2
    lm[10] = lm[9] * 0.7 + lm[13] * 0.3
    lm[14] = lm[13] * 0.7 + lm[17] * 0.3
    lm[18] = lm[17] * 0.5 + lm[0] * 0.5

    for i in range(5):
        tip_idx = FINGER_TIP[i]
        pip_idx = FINGER_PIP[i]
        mcp_idx = FINGER_MCP[i]
        mcp = lm[mcp_idx].copy()
        wrist = lm[0]

        if extended_fingers[i]:
            # 伸展：指尖向远离手腕方向
            # direction_away = mcp - wrist 指向远离手腕
            direction_away = mcp - wrist
            dir_len = float(np.linalg.norm(direction_away))
            if dir_len > 1e-6:
                direction_away /= dir_len
            # PIP 沿远离手腕方向偏移
            lm[pip_idx] = mcp + direction_away * 0.06
            # 指尖更远
            lm[tip_idx] = mcp + direction_away * 0.16
            # 拇指向左偏移
            if i == 0:
                lm[pip_idx][0] -= 0.03
                lm[tip_idx][0] -= 0.06
            # 小指向右偏移
            if i == 4:
                lm[pip_idx][0] += 0.03
                lm[tip_idx][0] += 0.06
        else:
            # 弯曲：指尖向手腕方向靠拢（不是零向量，而是靠近手腕）
            # dist(tip, wrist) / dist(mcp, wrist) ≈ 1.0-1.1  → 判定为弯曲
            direction_to_wrist = wrist - mcp
            dir_len = float(np.linalg.norm(direction_to_wrist))
            if dir_len > 1e-6:
                direction_to_wrist /= dir_len
            # PIP靠近MCP
            lm[pip_idx] = mcp.copy()
            # 指尖在MCP和手腕之间，靠近MCP一侧
            lm[tip_idx] = mcp + direction_to_wrist * 0.04

    # 确保所有点在有效范围
    lm[:, :2] = np.clip(lm[:, :2], 0.0, 1.0)

    return lm.astype(np.float32)


test_cases = [
    ("1", [False, True, False, False, False]),
    ("2", [False, True, True, False, False]),
    ("3", [False, True, True, True, False]),
    ("4", [False, True, True, True, True]),
    ("5", [True, True, True, True, True]),
]

all_pass = True

for expected, pattern in test_cases:
    lm = make_landmarks(pattern)
    flat = landmarks_to_flatten(lm)

    recognizer.clear()
    # 喂入足够帧以通过最小帧数门槛
    for _ in range(15):
        token = recognizer.classify_frame(flat, confidence_hint=0.8)

    diag2 = recognizer.get_diagnostic_info()
    tokens = diag2['tokens']

    if expected in tokens:
        print(f"  [OK] 手势'{expected}' -> 识别成功: {tokens}")
    else:
        print(f"  [FAIL] 手势'{expected}' -> 识别失败: tokens={tokens}")
        all_pass = False

# ==============================================
# 4. 完整模型链路验证
# ==============================================
print("\n[Step 4] 完整 RealSignLanguageModel 链路验证")
print("-" * 40)

model = RealSignLanguageModel()
model.load()
print(f"  [OK] RealSignLanguageModel 加载成功")

if model._recognizer:
    d = model._recognizer.get_diagnostic_info()
    print(f"  识别模式: {d['model_type']}")
    print(f"  词汇表前10: {d['vocabulary_first_10']}")

# 空白图像测试 (不崩溃即可)
img = np.ones((480, 640, 3), dtype=np.uint8) * 128
result = model.predict_frame(img)
print(f"  predict_frame: hands={len(result['hands_data'])}, tokens={result['tokens']}")
print("  [OK] 完整链路无异常")

model.unload()
detector.close()

# ==============================================
# 结果
# ==============================================
print("\n" + "=" * 60)
if all_pass:
    print("  RESULT: 全部验证通过! 数字手势 1-5 识别就绪")
else:
    print("  RESULT: 部分验证未通过，请检查日志")
print("=" * 60)
print("""
启动命令:
    cd d:/PythonProject/PythonProject/mute
    python backend/main.py
""")
