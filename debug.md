# 训练模型崩溃 Debug 记录

## 问题现象

训练 CSLTransformer 手语识别模型时，验证集准确率始终不涨，模型将所有样本预测为同一个类。

### 训练日志（崩溃状态）

```
Epoch  1: Train Acc=45.73%  Val Acc=3.23%
Epoch  5: Train Acc=56.22%  Val Acc=3.23%
Epoch 14: Train Acc=59.19%  Val Acc=3.23%  ← 纹丝不动
```

测试集混淆矩阵（5 类实验）：

```
        手语  对不起  很  你  在(预测)
手语      0     0    0   0   231
对不起    0     0    0   0   231
很        0     0    0   0   210
你        0     0    0   0   189
在        0     0    0   0   189    ← 全预测为"在"
```

26 类实验同样 —— 全预测为一个类。

## 排除过程

### 第一步：检查数据质量

- 所有 `.npy` 文件 shape 正确：`(90, 126)`，匹配 `CSL_INPUT_DIM=126`
- 无空数据、无 NaN、无 Inf，数值范围正常（MediaPipe 归一化坐标 0~1）
- 词汇表 26 类和磁盘目录一一对应
- 手部关键点在不同手势之间有可区分的差异

**结论：数据没问题。**

### 第二步：检查标签对应

- 词汇表 `vocabulary.json` 与磁盘目录名一致
- 每个手势的文件数量合理（20~65 个）
- 标签加载正确

**结论：标签没问题。**

### 第三步：调整滑动窗口

原始 stride=3，每个 90 帧录制产生 21 个窗口，相邻窗口只差 3 帧，高度相关。

改为 stride=15 → 每文件 5 个窗口。

**结果：无效，仍然崩溃。**

### 第四步：减少类别 + 关闭数据增强

用 5 个手势测试，关闭数据增强。

**结果：仍然全预测一个类。**

### 第五步：检查模型梯度

- 所有 337 万参数梯度正常流通，零梯度参数 = 0
- 模型 forward 无异常

**结论：模型架构和梯度没有问题。**

### 第六步：逐项排查训练配置

构建对照实验，逐个去掉训练配置项，用相同数据跑 100 步：

| 配置 | Train Acc | 预测种类 | 状态 |
|------|-----------|---------|------|
| 纯基线（全去掉） | 25.0% | 8 | 正常学习 |
| +class_weight | 6.3% | 5 | 显著变差 |
| +label_smoothing | 15.6% | 9 | 尚可 |
| +weight_decay | 6.3% | 7 | 显著变差 |
| 全部加上 | 28.1% | 12 | 短步数还行 |

短步数测试均未崩溃，但单加 class_weight 或 weight_decay 已经明显拉低准确率。

### 第七步：完整训练模拟

去掉 class_weight、label_smoothing、WeightedRandomSampler、oversampling、数据增强，用 warmup + 余弦退火完整训练 30 epoch：

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1 | 23.6% | 45.4% |
| 5 | 74.5% | 54.5% |
| 10 | 86.0% | **84.7%** |
| 20 | 93.3% | **88.3%** |
| 30 | 97.2% | **91.4%** |

最佳验证准确率 **93.0%**，26 个类全部有预测。

## 根因

五个配置叠加在一起导致训练崩溃：

1. **数据增强**（augment_sequence）—— 6 种增强随机组合，噪声 + 遮罩 + 缩放 + 平移 + 时间缩放 + dropout，过于激进
2. **class_weight** —— 少数类权重放大梯度，训练不稳定
3. **label_smoothing=0.1** —— 与 class_weight 叠加，损失信号变弱
4. **WeightedRandomSampler (replacement=True)** —— 重复采样导致某些样本被过度训练
5. **过采样 (Oversampling)** —— 人为复制少数类样本，进一步破坏数据分布

单个配置影响有限，但五个叠加互相放大，在长时间训练（数千步）后模型梯度方向偏斜，最终收敛到只预测一个类。

## 解决方案

简化训练配置：

```python
# 改前
train_dataset = GestureDataset(..., augment=True)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
# + WeightedRandomSampler + oversampling

# 改后
train_dataset = GestureDataset(..., augment=False)
criterion = nn.CrossEntropyLoss()
# 使用简单 shuffle DataLoader
```

去掉的配置及原因：
- **数据增强**：数据采集条件已包含自然变化（不同录制、不同角度），增强反而破坏关键点结构
- **class_weight + label_smoothing**：26 类样本量差距不大（最大 65 文件，最小 20 文件），不需要类别平衡
- **WeightedRandomSampler + oversampling**：shuffle DataLoader 已足够打乱顺序

## 关键教训

1. **先简化，再复杂**。训练配置应该从最简单的基线开始，确认能跑通后再逐步加东西
2. **对照实验比直觉有用**。最初怀疑数据有问题、怀疑模型架构有问题，结果都不是
3. **短步数测试看不出累积效应**。100 步测试全部正常，但 1000+ 步后配置叠加导致崩溃
