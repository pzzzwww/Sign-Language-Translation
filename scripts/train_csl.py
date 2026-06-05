"""
【知识点：深度学习训练管线】

这是模型训练的完整流程脚本。从原始 .npy 关键点数据出发，经过
数据加载 → 增强 → 分割 → 训练 → 评估 → 导出，得到可用的模型权重。

涉及知识点：
  - 滑动窗口数据切分：将变长序列切为固定长度训练样本
  - 数据增强 (Data Augmentation): 人为增加训练数据多样性，防止过拟合
  - 训练/验证分割：80% 训练、20% 验证，评估模型泛化能力
  - 类别权重 (Class Weight): 样本少的手势给更高权重，防止模型偏向多数类
  - 过采样 (Oversampling): 对弱势类别重复采样，进一步缓解类别不平衡
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
WINDOW_STRIDE = 3


# ======================================================================
# 数据加载
# ======================================================================

def load_gesture_data() -> tuple[list[np.ndarray], list[int], list[str]]:
    """
    加载所有采集的手势数据。

    Returns:
        sequences: [(30, 63), ...] 固定长度关键点序列
        labels: 对应的类别 ID 列表
        vocabulary: 词汇表列表
    """
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(
            f"词汇表未找到: {VOCAB_PATH}\n"
            "请先运行 python scripts/collect_data.py 采集数据"
        )

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_dict: dict[str, int] = json.load(f)

    # label_id → 手势名
    vocabulary = [""] * len(vocab_dict)
    for name, idx in vocab_dict.items():
        vocabulary[idx] = name

    sequences: list[np.ndarray] = []
    labels: list[int] = []

    for gesture_name, label_id in vocab_dict.items():
        gesture_dir = DATA_DIR / gesture_name
        npy_files = sorted(gesture_dir.glob("*.npy"))

        if not npy_files:
            logger.warning("手势 [%s] 无数据文件，跳过", gesture_name)
            continue

        for npy_file in npy_files:
            data = np.load(str(npy_file))  # (T, 63)

            if data.ndim != 2 or data.shape[1] != CSL_INPUT_DIM:
                logger.warning("跳过异常文件: %s (shape=%s, 期望维度=%d)",
                             npy_file.name, data.shape, CSL_INPUT_DIM)
                continue

            # 滑动窗口切分
            for start in range(0, data.shape[0] - SEQUENCE_LENGTH + 1, WINDOW_STRIDE):
                window = data[start:start + SEQUENCE_LENGTH]
                sequences.append(window)
                labels.append(label_id)

        logger.info("加载 [%s]: %d 文件 → %d 样本 (stride=%d)",
                    gesture_name, len(npy_files),
                    len(sequences) - (len(sequences) - len(labels) + 1 if sequences else 0),
                    WINDOW_STRIDE)

    # 重新统计
    n_per_class = {}
    for l in labels:
        n_per_class[vocabulary[l]] = n_per_class.get(vocabulary[l], 0) + 1

    logger.info("数据加载完成: %d 个样本, %d 类", len(sequences), len(vocabulary))
    for name, count in sorted(n_per_class.items()):
        logger.info("  %s: %d 样本", name, count)

    return sequences, labels, vocabulary


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
    sequences: list[np.ndarray],
    labels: list[int],
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
    from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
    from src.models.sign_language_model.csl_recognizer import CSLTransformer

    num_classes = len(vocabulary)

    # --- 数据分割 ---
    dataset = GestureDataset(sequences, labels, augment=False)
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train

    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    train_dataset.dataset._augment = True  # type: ignore[attr-defined]

    # ==== 类别权重（自动补偿样本不均衡）====
    # 【知识点：类别不平衡处理】如果"你好"有 2000 个样本而"谢谢"只有 200 个，
    # 模型会倾向预测"你好"。类别权重让少数类在损失函数中贡献更大。
    train_labels = [labels[i] for i in train_dataset.indices]
    class_counts = np.bincount(train_labels, minlength=num_classes)
    # 倒数加权 + 平滑（+1 防止除零）
    class_weights = 1.0 / (class_counts.astype(np.float32) + 1)
    # 归一化：权重平均 = num_classes
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    logger.info("类别权重 (越少数样本权重越高):")
    for i, (name, count, w) in enumerate(
        zip(vocabulary, class_counts, class_weights)
    ):
        logger.info("  %s: 样本=%d, 权重=%.2f", name, count, w)

    # ==== 过采样（Oversampling）====
    # 【知识点：过采样】如果某类样本数 < 平均值的 50%，就把该类样本重复放入训练集
    # 这样每个 batch 中弱势类的出现概率更高，缓解类别不平衡
    oversampled_indices = list(train_dataset.indices)
    mean_count = class_counts[class_counts > 0].mean()
    threshold = max(1, mean_count * 0.5)
    for class_id, count in enumerate(class_counts):
        if count > 0 and count < threshold:
            class_indices = [j for j, idx in enumerate(train_dataset.indices)  # type: ignore[attr-defined]
                           if labels[idx] == class_id]
            repeat = int(threshold / count) - 1
            oversampled_indices.extend(class_indices * repeat)
            logger.info("  过采样: %s %d→%d (+%d)",
                        vocabulary[class_id], count, count * (repeat + 1), count * repeat)

    # 用过采样后的索引重建 DataLoader
    train_sampler = WeightedRandomSampler(
        [class_weights[labels[i]] for i in oversampled_indices],
        num_samples=len(oversampled_indices),
        replacement=True,
    )
    train_dataset_oversampled = GestureDataset(
        [sequences[i] for i in oversampled_indices],
        [labels[i] for i in oversampled_indices],
        augment=True,
    )
    train_loader = DataLoader(train_dataset_oversampled, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info("过采样后训练集: %d 样本 | 验证集: %d 样本 | 批次大小: %d",
                len(oversampled_indices), n_val, batch_size)

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

    # --- 损失（类别加权 + 标签平滑） & 优化器 ---
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor, label_smoothing=0.1,
    )
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
    # 【知识点：混淆矩阵 Confusion Matrix】
    # 行 = 真实标签，列 = 预测标签
    # 对角线 = 预测正确，非对角线 = 混淆错误
    # 例：confusion[0,2]=5 → "你好"被误判为"对不起"5次
    # ==================================================================
    logger.info("\n=== 验证集混淆矩阵 (best model, 行=真实, 列=预测) ===")
    model.load_state_dict(best_model_state or {})
    model.eval()

    import itertools
    confusion = np.zeros((num_classes, num_classes), dtype=np.int32)

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            for true, pred in zip(y.cpu(), preds.cpu()):
                confusion[true.item(), pred.item()] += 1

    # 打印每类准确率 + 最常混淆的类别
    for i in range(num_classes):
        total_i = confusion[i].sum()
        correct_i = confusion[i, i]
        acc = correct_i / total_i * 100 if total_i > 0 else 0

        # 找出该类最常被误判成的类别（排除对角线）
        mistakes = [(confusion[i, j], vocabulary[j])
                   for j in range(num_classes) if j != i and confusion[i, j] > 0]
        mistakes.sort(reverse=True)

        status = "✓" if acc >= 85 else ("⚠" if acc >= 60 else "✗")
        line = f"  {status} {vocabulary[i]}: {acc:.1f}% ({correct_i}/{total_i})"
        if mistakes:
            top_3 = mistakes[:3]
            line += "  混淆→ " + ", ".join(
                f"{name}({count})" for count, name in top_3
            )
        logger.info(line)

    # 打印完整混淆矩阵表格
    logger.info("\n--- 完整混淆矩阵 ---")
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

    # 1. 加载数据
    sequences, labels, vocabulary = load_gesture_data()

    if len(sequences) < 10:
        logger.error(
            "训练数据不足 (仅 %d 个样本)。请先采集更多数据: python scripts/collect_data.py",
            len(sequences),
        )
        return

    # 2. 训练
    train(
        sequences=sequences,
        labels=labels,
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
