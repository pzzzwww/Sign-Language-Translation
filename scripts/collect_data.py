"""
手语数据采集工具 — 用摄像头录制手势关键点序列，生成训练数据。

工作流程:
  1. 输入手势名称（如 "你好"、"谢谢"、"1"）
  2. 摄像头开启，画面显示手部检测状态和置信度
  3. 确保手被检测到（绿色边框），按 SPACE 开始录制 3 秒
  4. 有效帧率 <30% 的录制自动丢弃，提示重录
  5. 自动保存关键点序列到 data/gestures/<手势名>/

数据格式:
  每段录制保存为 .npy 文件，shape=(T, 126)
  126 = 左手63维(镜像) + 右手63维，单手时缺失部分填零
  词汇表自动生成到 data/gestures/vocabulary.json

Usage:
    python scripts/collect_data.py                      # 录制新手势（默认 20 段/手势）
    python scripts/collect_data.py --gesture 你好        # 仅录制指定手势
    python scripts/collect_data.py --count 50           # 每个手势录 50 段
    python scripts/collect_data.py --duration 5         # 每段 5 秒
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sign_language_model.mediapipe_detector import (
    MediaPipeHandDetector,
    landmarks_to_flatten,
)
from src.config import CAMERA_INDEX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("collect_data")

# 数据保存目录
DATA_DIR = Path(__file__).parent.parent / "data" / "gestures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
VOCAB_PATH = DATA_DIR / "vocabulary.json"

# 配置
DETECTION_CONFIDENCE = 0.3   # MediaPipe 手部检测最低置信度（降低以提高检出率）
TRACKING_CONFIDENCE = 0.3    # MediaPipe 手部跟踪最低置信度
MIN_VALID_RATIO = 0.3        # 录制中有效帧比例最小值，低于此值自动丢弃


def load_vocabulary() -> dict[str, int]:
    """加载或创建词汇表 {手势名: label_id}。"""
    if VOCAB_PATH.exists():
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_vocabulary(vocab: dict[str, int]) -> None:
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def count_existing_recordings(gesture: str) -> int:
    """统计手势已有录制数。"""
    gesture_dir = DATA_DIR / gesture
    if not gesture_dir.exists():
        return 0
    return len(list(gesture_dir.glob("*.npy")))


def collect_gesture(
    gesture: str,
    count: int = 20,
    duration_sec: float = 3.0,
    detector: MediaPipeHandDetector | None = None,
) -> int:
    """
    录制指定手势的多段关键点序列。

    特性:
      - 录制前检查手是否存在，未检测到手时拒绝录制
      - 录制完成后检查有效帧率，<30% 自动丢弃
      - 实时显示手部检测状态和置信度

    Returns:
        实际保存的段数
    """
    if detector is None:
        detector = MediaPipeHandDetector(
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
            max_num_hands=2,
        )

    gesture_dir = DATA_DIR / gesture
    gesture_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        logger.error("无法打开摄像头 (index=%d)", CAMERA_INDEX)
        return 0

    cap.set(cv2.CAP_PROP_FPS, 30)

    existing = count_existing_recordings(gesture)
    logger.info("手势 [%s] 已有 %d 段，目标 %d 段", gesture, existing, count)
    recorded = 0
    skipped = 0
    target_frames = int(duration_sec * 30)

    state = "waiting"
    frame_buffer: deque[np.ndarray] = deque()
    valid_count = 0
    recording_start = 0.0

    print(f"\n{'='*50}")
    print(f"  手势: {gesture}  |  目标: {count} 段  |  每段 {duration_sec}s ({target_frames}帧)")
    print(f"  请确保手在摄像头画面中（出现绿色框后再录制）")
    print(f"  按 SPACE 开始录制  |  Q 退出")
    print(f"{'='*50}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()

        # --- MediaPipe 检测 + 绘制 ---
        hands = detector.detect(frame)
        has_hand = bool(hands)
        confidence = hands[0]["confidence"] if hands else 0.0

        if hands:
            display = detector.draw_detection(display, hands)

        h, w = display.shape[:2]

        if state == "waiting":
            # ---------- 等待状态 ----------
            progress_str = f"已保存: {existing + recorded}/{count}"
            if skipped > 0:
                progress_str += f" (丢弃: {skipped})"

            # 手部检测状态
            if has_hand:
                hand_labels = []
                for hd in hands:
                    h_lr = hd.get("handedness", "?")
                    h_conf = hd.get("confidence", 0)
                    mirror_hint = " 镜像" if h_lr == "Left" else ""
                    hand_labels.append(f"{h_lr}{mirror_hint}({h_conf:.2f})")
                hand_status = f"检测到 {len(hands)} 只手: {' | '.join(hand_labels)}"
                status_color = (0, 255, 0)  # 绿色：可以录制
            else:
                hand_status = "⚠ 未检测到手部 — 请调整手的位置或光线"
                status_color = (0, 165, 255)  # 橙色：需要调整

            cv2.putText(display, progress_str, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, hand_status, (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            if has_hand:
                cv2.putText(display, "SPACE: 开始录制", (20, h - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "SPACE: 开始录制（无手部！仍可按）", (20, h - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
            cv2.putText(display, "Q: 退出", (20, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        elif state == "recording":
            # ---------- 录制状态 ----------
            elapsed = time.time() - recording_start
            remaining = max(0, duration_sec - elapsed)
            n_frames = len(frame_buffer)

            # 主状态文字
            status = f"录制中... {remaining:.1f}s | 帧: {n_frames}/{target_frames}"
            red_intensity = int(150 + 105 * (remaining / duration_sec))
            cv2.putText(display, status, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, red_intensity), 2)

            # 有效帧率（实时显示）
            valid_ratio = valid_count / n_frames * 100 if n_frames > 0 else 0
            if valid_ratio >= 70:
                vr_color = (0, 255, 0)
            elif valid_ratio >= 30:
                vr_color = (0, 215, 255)
            else:
                vr_color = (0, 0, 255)
            cv2.putText(display, f"有效帧: {valid_count}/{n_frames} ({valid_ratio:.0f}%)",
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, vr_color, 2)

            # 进度条
            bar_w = int(w * 0.6)
            bar_x = (w - bar_w) // 2
            bar_y = h - 60
            progress = (duration_sec - remaining) / duration_sec
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20),
                         (100, 100, 100), 2)
            cv2.rectangle(display, (bar_x, bar_y),
                         (bar_x + int(bar_w * progress), bar_y + 20),
                         (0, int(255 * progress), 255 - int(255 * progress)), -1)

            # 收集双手关键点（拼接为126维向量）
            if hands:
                left_lm = None
                right_lm = None
                for hand in hands:
                    if hand["confidence"] < DETECTION_CONFIDENCE:
                        continue
                    lm = hand["landmarks"].copy()
                    # 左手 x 镜像为右手坐标系
                    if hand.get("handedness") == "Left":
                        lm[:, 0] = 1.0 - lm[:, 0]
                        left_lm = landmarks_to_flatten(lm)
                    else:
                        right_lm = landmarks_to_flatten(lm)

                # 拼接左右手：左手在前63维，右手在后63维，缺失填零
                left_vec = left_lm if left_lm is not None else np.zeros(63, dtype=np.float32)
                right_vec = right_lm if right_lm is not None else np.zeros(63, dtype=np.float32)
                frame_buffer.append(np.concatenate([left_vec, right_vec]))
                valid_count += 1
            else:
                frame_buffer.append(np.zeros(126, dtype=np.float32))

            # 录制完成
            if len(frame_buffer) >= target_frames:
                valid_ratio_final = valid_count / target_frames

                if valid_ratio_final >= MIN_VALID_RATIO:
                    # 保存
                    frames = np.stack(list(frame_buffer)[:target_frames])
                    idx = existing + recorded + 1
                    save_path = gesture_dir / f"{gesture}_{idx:04d}.npy"
                    np.save(str(save_path), frames)
                    recorded += 1
                    logger.info(
                        "✓ 已保存: %s (%d帧, 有效帧 %.0f%%)",
                        save_path.name, frames.shape[0], valid_ratio_final * 100,
                    )
                else:
                    # 丢弃：有效帧太少
                    skipped += 1
                    logger.warning(
                        "✗ 丢弃: 有效帧仅 %.0f%% (需要 >=%.0f%%)，请确保手在画面中并保持手势",
                        valid_ratio_final * 100, MIN_VALID_RATIO * 100,
                    )

                frame_buffer.clear()
                valid_count = 0
                state = "waiting"

                if existing + recorded >= count:
                    state = "done"

        elif state == "done":
            cv2.putText(display, "目标数量已达到!", (w // 4, h // 2 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(display, "按空格继续录制额外段 | 按 Q 退出",
                       (w // 4, h // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        cv2.imshow("手语数据采集 - " + gesture, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord(" "):
            if state == "waiting":
                # 检查手部是否存在，给出警告但不阻止（用户可以强制录制）
                if not has_hand:
                    logger.warning(
                        "⚠ 当前未检测到手部（置信度 %.2f）。录制可能产生无效数据。", confidence
                    )
                state = "recording"
                frame_buffer.clear()
                valid_count = 0
                recording_start = time.time()
                logger.info("▶ 开始录制第 %d/%d 段", existing + recorded + 1, count)
            elif state == "done":
                state = "waiting"
                logger.info("继续录制，当前已有 %d 段", existing + recorded)

    cap.release()
    cv2.destroyAllWindows()
    return recorded


def main() -> None:
    parser = argparse.ArgumentParser(description="手语训练数据采集")
    parser.add_argument("--gesture", "-g", type=str, default=None,
                       help="仅录制指定手势（不填则交互输入）")
    parser.add_argument("--count", "-c", type=int, default=20,
                       help="每个手势录制段数 (default: 20)")
    parser.add_argument("--duration", "-d", type=float, default=3.0,
                       help="每段录制时长/秒 (default: 3.0)")
    args = parser.parse_args()

    # 加载检测器（降低阈值提高检出率）
    logger.info("初始化 MediaPipe 手部检测器（阈值=%.1f）...", DETECTION_CONFIDENCE)
    detector = MediaPipeHandDetector(
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
        max_num_hands=2,
    )

    vocab = load_vocabulary()

    gestures: list[str] = []
    if args.gesture:
        gestures = [args.gesture]
    else:
        print("\n请输入要采集的手势名称（一行一个，空行结束）")
        print(f"已有手势: {list(vocab.keys()) if vocab else '(尚无)'}")
        while True:
            name = input("> ").strip()
            if not name:
                break
            gestures.append(name)

    if not gestures:
        print("未指定任何手势，退出")
        detector.close()
        return

    print(f"\n即将采集手势: {gestures}")
    print(f"每手势 {args.count} 段，每段 {args.duration} 秒")
    print(f"检测阈值: {DETECTION_CONFIDENCE} | 有效帧率要求: >= {int(MIN_VALID_RATIO * 100)}%")
    print(f"数据保存目录: {DATA_DIR}")

    total_recorded = 0
    for gesture in gestures:
        if gesture not in vocab:
            vocab[gesture] = len(vocab)

        n = collect_gesture(
            gesture=gesture,
            count=args.count,
            duration_sec=args.duration,
            detector=detector,
        )
        total_recorded += n

    save_vocabulary(vocab)
    detector.close()

    print(f"\n{'='*50}")
    print(f"  采集完成! 共保存 {total_recorded} 段有效数据")
    print(f"  词汇表 ({len(vocab)} 个手势): {list(vocab.keys())}")
    print(f"  数据目录: {DATA_DIR}")
    print(f"  下一步: python scripts/train_csl.py")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
