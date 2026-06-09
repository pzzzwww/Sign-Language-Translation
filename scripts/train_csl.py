"""
【知识点：深度学习训练管线】

这是模型训练的完整流程脚本。从原始 .npy 关键点数据出发，经过
数据加载 → 增强 → 分割 → 训练 → 评估 → 导出，得到可用的模型权重。

涉及知识点：
  - 滑动窗口数据切分：将变长序列切为固定长度训练样本
  - 数据增强 (Data Augmentation): 人为增加训练数据多样性，防止过拟合
  - 文件级数据分割：70% 训练 / 15% 验证 / 15% 测试，防止同一录制数据泄露
  - AdamW 优化器: Adam + 解耦权重衰减，当前主流优化器
  - 余弦退火 (Cosine Annealing): 学习率随训练逐渐降低
  - 早停 (Early Stopping): 验证准确率不提升就停，防止过拟合
  - 混淆矩阵 (Confusion Matrix): 看模型在哪些类别之间容易搞混
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CSL_INPUT_DIM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_csl")

# 路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "gestures"
PRETRAINED_DIR = PROJECT_ROOT / "src" / "models" / "sign_language_model" / "pretrained"
VOCAB_PATH = DATA_DIR / "vocabulary.json"

# 序列长度（帧数）
SEQUENCE_LENGTH = 30
# 滑动窗口步长（帧）
WINDOW_STRIDE = 15


# ======================================================================
# 数据加载
# ======================================================================

def load_gesture_data() -> tuple[list[tuple[np.ndarray, int]], list[str]]:
    """
    加载所有采集的手势数据（文件级别，不做滑动窗口切分）。

    Returns:
        file_data: [(npy_array, label_id), ...] 每个 .npy 文件一个条目
        vocabulary: 词汇表列表
    """
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(
            f"词汇表未找到: {VOCAB_PATH}\n"
            "请先运行 python scripts/collect_data.py 采集数据"
        )

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_dict: dict[str, int] = json.load(f)

    vocabulary = [""] * len(vocab_dict)
    for name, idx in vocab_dict.items():
        vocabulary[idx] = name

    file_data: list[tuple[np.ndarray, int]] = []

    for gesture_name, label_id in vocab_dict.items():
        gesture_dir = DATA_DIR / gesture_name
        npy_files = sorted(gesture_dir.glob("*.npy"))

        if not npy_files:
            logger.warning("手势 [%s] 无数据文件，跳过", gesture_name)
            continue

        for npy_file in npy_files:
            data = np.load(str(npy_file))

            if data.ndim != 2 or data.shape[1] != CSL_INPUT_DIM:
                logger.warning("跳过异常文件: %s (shape=%s, 期望维度=%d)",
                             npy_file.name, data.shape, CSL_INPUT_DIM)
                continue

            file_data.append((data, label_id))

        logger.info("加载 [%s]: %d 个文件", gesture_name, len(npy_files))

    logger.info("数据加载完成: %d 个文件, %d 类", len(file_data), len(vocabulary))
    return file_data, vocabulary


# ======================================================================
# 数据增强
# 【知识点：数据增强】对训练数据做微小扰动，增加多样性，减少过拟合
# ======================================================================

def augment_sequence(seq: np.ndarray, noise_std: float = 0.04, mask_prob: float = 0.15) -> np.ndarray:
    """
    对关键点序列做数据增强，模拟各种真实场景的变化。

    增强策略（组合使用，每次随机选 2-3 种）：
      1. 高斯噪声 — 模拟 MediaPipe 检测抖动
      2. 时间遮罩 — 模拟短暂遮挡/检测丢失
      3. 空间缩放 — 模拟手离摄像头远近变化
      4. 空间平移 — 模拟手在画面中不同位置
      5. 时间缩放 — 模拟手势快慢变化（不同人做手势速度不同）
      6. 坐标 dropout — 模拟部分关键点检测失败
    """
    augmented = seq.copy()
    T, D = augmented.shape

    # 收集可用增强方法
    candidates = []

    # 1. 高斯噪声（始终应用）
    noise_level = np.random.uniform(0.01, noise_std)
    augmented += np.random.randn(T, D).astype(np.float32) * noise_level

    # 2. 时间遮罩
    if np.random.random() < 0.6:
        mask_len = np.random.randint(1, max(2, int(T * mask_prob)))
        mask_start = np.random.randint(0, T - mask_len)
        augmented[mask_start:mask_start + mask_len] *= np.random.uniform(0, 0.3)

    # 3. 空间缩放（每手 63 维 = 21 关键点 × 3 坐标）
    if np.random.random() < 0.5 and D >= 63:
        scale = np.random.uniform(0.8, 1.2)
        for hand_offset in [0, 63]:  # 左手、右手
            if hand_offset + 63 <= D:
                hand = augmented[:, hand_offset:hand_offset + 63].reshape(T, 21, 3)
                center = hand.mean(axis=(0, 1), keepdims=True)
                hand = (hand - center) * scale + center
                augmented[:, hand_offset:hand_offset + 63] = hand.reshape(T, 63)

    # 4. 空间平移
    if np.random.random() < 0.5 and D >= 63:
        shift = np.random.uniform(-0.05, 0.05, size=(1, 3))
        for hand_offset in [0, 63]:
            if hand_offset + 63 <= D:
                hand = augmented[:, hand_offset:hand_offset + 63].reshape(T, 21, 3)
                hand += shift
                augmented[:, hand_offset:hand_offset + 63] = hand.reshape(T, 63)

    # 5. 时间缩放（插值模拟快慢变化）
    if np.random.random() < 0.4:
        speed = np.random.uniform(0.7, 1.3)
        new_T = max(5, int(T * speed))
        indices = np.linspace(0, T - 1, new_T)
        scaled = np.zeros((T, D), dtype=np.float32)
        for d in range(D):
            scaled[:, d] = np.interp(np.arange(T), np.arange(new_T) * (T - 1) / (new_T - 1) if new_T > 1 else np.arange(new_T),
                                    augmented[np.clip(indices.astype(int), 0, T - 1), d])
        augmented = scaled

    # 6. 坐标 dropout（随机丢弃部分关键点）
    if np.random.random() < 0.3 and D >= 63:
        n_drop = np.random.randint(1, 5)  # 丢弃 1-4 个关键点
        for _ in range(n_drop):
            lm_idx = np.random.randint(0, 21)  # 随机选一个关键点
            start = lm_idx * 3
            for hand_offset in [0, 63]:
                if hand_offset + start + 3 <= D:
                    augmented[:, hand_offset + start:hand_offset + start + 3] *= 0.1

    return augmented


# ======================================================================
# 数据集
# ======================================================================

class GestureDataset:
    """手语关键点序列数据集。"""

    def __init__(self, sequences: list[np.ndarray], labels: list[int], augment: bool = False):
        self._sequences = sequences
        self._labels = labels
        self._augment = augment

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> tuple:
        import torch
        seq = self._sequences[idx].copy()

        if self._augment:
            seq = augment_sequence(seq)

        x = torch.from_numpy(seq).float()
        y = torch.tensor(self._labels[idx], dtype=torch.long)
        return x, y


# ======================================================================
# 训练
# ======================================================================

def train(
    file_data: list[tuple[np.ndarray, int]],
    vocabulary: list[str],
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    patience: int = 25,
) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from src.models.sign_language_model.csl_recognizer import CSLTransformer

    num_classes = len(vocabulary)

    # --- 逐类分层数据分割：70% 训练 / 15% 验证 / 15% 测试 ---
    # 每个手势内部洗牌后按比例分配，确保每类在三个集合中都有代表
    rng = np.random.default_rng(42)
    train_files: list[tuple[np.ndarray, int]] = []
    val_files: list[tuple[np.ndarray, int]] = []
    test_files: list[tuple[np.ndarray, int]] = []

    for label_id in range(num_classes):
        class_files = [(d, l) for d, l in file_data if l == label_id]
        n = len(class_files)
        if n == 0:
            continue
        idx = rng.permutation(n)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        train_files.extend(class_files[i] for i in idx[:n_train])
        val_files.extend(class_files[i] for i in idx[n_train:n_train + n_val])
        test_files.extend(class_files[i] for i in idx[n_train + n_val:])

    logger.info("逐类分层划分: 训练 %d / 验证 %d / 测试 %d (共 %d 个文件)",
                len(train_files), len(val_files), len(test_files), len(file_data))

    def _make_windows(
        files: list[tuple[np.ndarray, int]],
    ) -> tuple[list[np.ndarray], list[int]]:
        seqs: list[np.ndarray] = []
        lbls: list[int] = []
        for data, label in files:
            for start in range(0, data.shape[0] - SEQUENCE_LENGTH + 1, WINDOW_STRIDE):
                seqs.append(data[start:start + SEQUENCE_LENGTH])
                lbls.append(label)
        return seqs, lbls

    train_sequences, train_labels = _make_windows(train_files)
    val_sequences, val_labels = _make_windows(val_files)
    test_sequences, test_labels = _make_windows(test_files)

    logger.info("窗口数: 训练 %d | 验证 %d | 测试 %d",
                len(train_sequences), len(val_sequences), len(test_sequences))

    train_dataset = GestureDataset(train_sequences, train_labels, augment=False)
    val_dataset = GestureDataset(val_sequences, val_labels, augment=False)
    test_dataset = GestureDataset(test_sequences, test_labels, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info("训练集: %d 样本 | 验证集: %d 样本 | 测试集: %d 样本 | 批次大小: %d",
                len(train_sequences), len(val_labels), len(test_labels), batch_size)

    # --- 模型 ---
    model = CSLTransformer(
        num_classes=num_classes,
        input_dim=CSL_INPUT_DIM,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("模型参数量: %d", n_params)

    # --- 损失 & 优化器 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    # 余弦退火 + 线性 warmup（前 5 个 epoch 学习率从 0 线性增长）
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - 5, eta_min=lr * 0.01,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5),
            scheduler_cosine,
        ],
        milestones=[5],
    )

    # ==================================================================
    # 【知识点：训练循环 Train Loop】
    # 每个 epoch = 完整遍历一次训练集
    #   → 前向传播 (model(x) 预测)
    #   → 损失计算 (预测 vs 真实标签)
    #   → 反向传播 (loss.backward() 计算梯度)
    #   → 参数更新 (optimizer.step() 按梯度调整权重)
    # 验证阶段不计算梯度 (torch.no_grad())，纯粹评估模型
    # ==================================================================
    best_val_acc = 0.0
    best_model_state: dict | None = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # -- 训练阶段 --
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)

        train_acc = train_correct / train_total if train_total > 0 else 0
        train_loss /= train_total

        # -- 验证阶段 --
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                val_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0
        val_loss /= val_total

        scheduler.step()

        # 日志
        logger.info(
            "Epoch %3d | Train Loss: %.4f Acc: %.2f%% | Val Loss: %.4f Acc: %.2f%% | LR: %.2e",
            epoch, train_loss, train_acc * 100, val_loss, val_acc * 100,
            scheduler.get_last_lr()[0],
        )

        # 早停 & 保存最佳
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("早停: 验证准确率 %d 轮未提升", patience)
                break

    # --- 保存模型 ---
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    if best_model_state is not None:
        model_path = PRETRAINED_DIR / "csl_model.pt"
        torch.save(best_model_state, str(model_path))
        logger.info("模型权重已保存: %s (Val Acc: %.2f%%)", model_path, best_val_acc * 100)
    else:
        logger.warning("未找到最佳模型，可能是训练失败")

    # --- 保存词汇表 ---
    vocab_path = PRETRAINED_DIR / "csl_vocabulary.json"
    vocab_dict = {name: idx for idx, name in enumerate(vocabulary)}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    logger.info("词汇表已保存: %s (%d 个手势)", vocab_path, len(vocabulary))

    # ==================================================================
    # 测试集评估（held-out，未参与训练/验证）
    # ==================================================================
    logger.info("\n" + "=" * 50)
    logger.info("=== 测试集评估 (held-out) ===")
    logger.info("=" * 50)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()

    confusion = np.zeros((num_classes, num_classes), dtype=np.int32)
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            test_correct += (preds == y).sum().item()
            test_total += y.size(0)
            for true, pred in zip(y.cpu(), preds.cpu()):
                confusion[true.item(), pred.item()] += 1

    test_acc = test_correct / test_total * 100 if test_total > 0 else 0
    logger.info("测试集总体准确率: %.2f%% (%d/%d)", test_acc, test_correct, test_total)

    # 每类 Precision / Recall / F1
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)
    f1_per_class = np.zeros(num_classes)
    for i in range(num_classes):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        precision_per_class[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_per_class[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_per_class[i] = (2 * precision_per_class[i] * recall_per_class[i]
                           / (precision_per_class[i] + recall_per_class[i])
                           if (precision_per_class[i] + recall_per_class[i]) > 0 else 0.0)

    macro_f1 = f1_per_class.mean() * 100
    logger.info("宏平均 F1: %.2f%% | 宏平均 Precision: %.2f%% | 宏平均 Recall: %.2f%%",
                macro_f1, precision_per_class.mean() * 100, recall_per_class.mean() * 100)

    logger.info("\n--- 各类指标 ---")
    for i in range(num_classes):
        total_i = confusion[i].sum()
        correct_i = confusion[i, i]
        acc = recall_per_class[i] * 100

        mistakes = [(confusion[i, j], vocabulary[j])
                   for j in range(num_classes) if j != i and confusion[i, j] > 0]
        mistakes.sort(reverse=True)

        status = "✓" if acc >= 85 else ("⚠" if acc >= 60 else "✗")
        line = (f"  {status} {vocabulary[i]}: Acc={acc:.1f}% P={precision_per_class[i]:.2f} "
                f"R={recall_per_class[i]:.2f} F1={f1_per_class[i]:.2f} "
                f"({correct_i}/{total_i})")
        if mistakes:
            top_3 = mistakes[:3]
            line += "  混淆→ " + ", ".join(
                f"{name}({count})" for count, name in top_3
            )
        logger.info(line)

    logger.info("\n--- 完整混淆矩阵 (行=真实, 列=预测) ---")
    col_width = max(len(name) for name in vocabulary) + 1
    header = " " * col_width + "".join(f"{name:>6}" for name in vocabulary)
    logger.info(header)
    for i, name in enumerate(vocabulary):
        row = f"{name:<{col_width}}" + "".join(
            f"{confusion[i, j]:6d}" for j in range(num_classes)
        )
        logger.info(row)


# ======================================================================
# 入口
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="CSLTransformer 手语识别模型训练")
    parser.add_argument("--epochs", type=int, default=60, help="训练轮数 (default: 60)")
    parser.add_argument("--batch", type=int, default=16, help="批次大小 (default: 16)")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率 (default: 5e-4)")
    parser.add_argument("--device", type=str, default=None,
                       help="训练设备，不指定则自动检测 (cuda/cpu)")
    parser.add_argument("--patience", type=int, default=25,
                       help="早停耐心轮数 (default: 25)")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("CSLTransformer 手语识别模型训练")
    logger.info("=" * 50)

    # 自动检测设备
    if args.device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            logger.info("检测到 GPU: %s", torch.cuda.get_device_name(0))
        else:
            logger.info("未检测到 GPU，使用 CPU 训练")
    else:
        device = args.device

    logger.info("训练设备: %s", device)

    # 1. 加载数据（文件级别，不做窗口切分）
    file_data, vocabulary = load_gesture_data()

    if len(file_data) < 10:
        logger.error(
            "训练数据不足 (仅 %d 个文件)。请先采集更多数据: python scripts/collect_data.py",
            len(file_data),
        )
        return

    # 2. 训练（内部做文件级划分 + 滑动窗口）
    train(
        file_data=file_data,
        vocabulary=vocabulary,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=device,
        patience=args.patience,
    )

    logger.info("\n训练完成! 下一步:")
    logger.info("  1. 权重已保存到 src/models/sign_language_model/pretrained/csl_model.pt")
    logger.info("  2. 重启服务: python -m src.backend.main")
    logger.info("  3. 系统将自动加载 Transformer 权重替代启发式规则")


if __name__ == "__main__":
    main()
