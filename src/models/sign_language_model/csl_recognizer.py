"""
【知识点：CSL 手语识别器 — Transformer 时序分类】

这是整个项目的算法核心。把一段手部关键点序列（N帧×63维）输入 Transformer，
输出这段序列代表哪个手势（分类）。

涉及知识点：
  - Transformer Encoder: 谷歌2017年提出的注意力架构，本项目的核心算法
  - 自注意力 (Self-Attention): 让模型能"关注"序列中任意两帧之间的关系
  - 位置编码 (Positional Encoding): Transformer 本身不感知顺序，需要注入位置信息
  - 混合池化 (Mean+Max Pooling): 把变长序列压缩为固定长度向量用于分类
  - 降级策略: 模型权重缺失时自动切换启发式规则，保证系统不崩溃

架构: Input Projection → Positional Encoding → N×TransformerEncoderLayer → Mean+Max Pooling → Classifier
输入: T 帧 × 63 维关键点 (21 landmarks × 3)
输出: Token 标签 + 置信度
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import CSL_INPUT_DIM, CSL_TOKEN_BLACKLIST

logger = logging.getLogger(__name__)

# 【知识点：依赖检查】PyTorch 是可选依赖，未安装时降级到启发式模式
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


# ======================================================================
# 模型定义 — Transformer 架构
# ======================================================================

class PositionalEncoding(nn.Module):
    """【知识点：位置编码 Positional Encoding】

    Transformer 的自注意力是"无序"的 — 它不知道第1帧和第30帧的先后关系。
    位置编码给每帧加上一个可学习的向量，让模型知道"这是第几帧"。

    这里用的是可学习位置编码（Learnable PE）：
      - 随机初始化一个 (1, max_len, d_model) 的参数矩阵
      - 训练时自动学习最优的位置表示
      - 相比正弦位置编码更灵活，但需要训练数据支持
    """

    def __init__(self, d_model: int, max_len: int = 64) -> None:
        super().__init__()
        # 【知识点：nn.Parameter】告诉 PyTorch 这是一个可训练的参数，会被优化器更新
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model))
        # 【知识点：trunc_normal_ 初始化】截断正态分布，避免极端值
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # 加法：x 是输入特征，pe 是位置信息，相加即"注入位置"
        return x + self.pe[:, :x.size(1), :]


class CSLTransformer(nn.Module):
    """【知识点：CSL Transformer 模型架构】

    这是自定义的手语识别模型。把它拆开来看就是 4 个组件：

    1. Input Projection (线性映射):  63维  → 128维，升维方便后续处理
    2. Positional Encoding:         注入"第几帧"的位置信息
    3. Transformer Encoder:         4 层自注意力，让模型"理解"帧间关系
    4. Classifier:                  MLP 分类头，1280维 → 逐步降维 → 手势类别数

    关键设计决策：
      - d_model=128 小模型，CPU 友好；batch_first=True 输入格式为 (batch, seq, dim)
      - norm_first=True (Pre-LN): 层归一化在注意力之前，训练更稳定
      - GELU 激活: 比 ReLU 更平滑，在小模型上表现略好
      - Mean+Max Pooling: 同时保留全局趋势（均值）和局部峰值（最大值）

    参数量约 80 万，远小于 ViT-B/16 的 8600 万，CPU 也能快速推理。

    Args:
        num_classes: 手势类别数（训练词表大小）
        input_dim: 输入特征维度（126=双手关键点，894=+ViT）
        d_model: Transformer 隐层维度
        nhead: 多头注意力的头数（必须能被 d_model 整除）
        num_layers: Transformer Encoder 层数（深度）
        dim_feedforward: 前馈网络隐层维度（通常 = d_model × 4）
        dropout: 随机失活率，防止过拟合
    """

    def __init__(
        self,
        num_classes: int = 100,
        input_dim: int = 63,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ---- 1. 输入投影：将原始关键点向量映射到 Transformer 工作空间 ----
        # 【知识点：线性层 nn.Linear】y = xW^T + b，最简单的神经网络层
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        # ---- 2. Transformer Encoder ----
        # 【知识点：TransformerEncoderLayer】
        # 每个层 = 自注意力 + 前馈网络 + 残差连接 + 层归一化
        # 自注意力公式: Attention(Q,K,V) = softmax(QK^T/√d_k)V
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,                  # 8个头并行做注意力，捕获不同角度的关系
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",           # GELU 比 ReLU 平滑，训练梯度更稳
            batch_first=True,            # 输入格式 (batch, seq, dim)，更直观
            norm_first=True,             # Pre-LN: 先归一化再做注意力（训练更稳定）
        )
        # 【知识点：堆叠 Encoder】4 层 = 4 个 EncoderLayer 串联，层数越深感受野越大
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # ---- 3. 分类头：全连接 MLP ----
        # 【知识点：混合池化】mean 看全局趋势（手型整体），max 抓峰值（关键帧）
        # d_model*2 = 128*2 = 256，因为拼接了 mean 和 max 两个特征
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.LayerNorm(256),            # 层归一化：稳定训练
            nn.GELU(),
            nn.Dropout(dropout),          # 随机丢弃神经元，防止过拟合
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),    # 越深层 dropout 越小
            nn.Linear(128, num_classes),  # 最后一层输出 = 类别数
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """前向传播（模型推理的一趟完整流程）。

        Args:
            x: (B, T, D)
               B = batch_size 批次大小
               T = sequence_length 帧数
               D = input_dim 每帧特征维度

        Returns:
            (B, num_classes) 未归一化的分类分数（logits），softmax 后变概率
        """
        x = self.input_proj(x)                    # (B, T, d_model)  线性映射
        x = self.pos_encoding(x)                  # 注入位置信息
        x = self.encoder(x)                       # 自注意力时序建模
        # 池化：把变长序列 (B,T,d) 压缩为 (B,d*2)
        # 【知识点：torch.cat】沿指定维度拼接两个张量
        x = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=1)
        return self.classifier(x)                 # Softmax 在外层调用


# ======================================================================
# 词汇表
# ======================================================================

# 默认中文手语词汇表（可扩展，训练时会被替换）
DEFAULT_CSL_VOCABULARY = [
    # ★ 常用手势（启发式模式核心，按优先级排序）
    "你", "我", "点赞", "我爱你", "讨厌",
    "打电话", "OK", "摇滚", "胜利", "手枪", "牛",
    # 数字手势
    "1", "2", "3", "4", "5",
    # 基础代词
    "他", "她", "我们", "你们", "他们",
    # 常用动词
    "是", "有", "要", "想", "能", "会", "可以",
    "喜欢", "爱", "知道", "觉得", "说", "看", "听",
    "吃", "喝", "去", "来", "走", "跑", "坐",
    "给", "拿", "做", "用", "让", "叫", "告诉",
    # 常用名词
    "人", "朋友", "家", "学校", "医院", "商店", "钱",
    "时间", "今天", "明天", "昨天", "现在", "上午", "下午",
    "东西", "问题", "办法", "事情", "名字", "地方", "水",
    # 形容词
    "好", "坏", "大", "小", "多", "少", "快", "慢",
    "高兴", "难过", "累", "忙", "漂亮", "对", "错",
    # 疑问词
    "什么", "谁", "哪", "怎么", "为什么", "多少", "几",
    # 连词/助词
    "和", "的", "了", "吗", "呢", "吧", "也",
    "不", "很", "都", "还", "就", "才", "没",
    # 扩展词汇
    "谢谢", "对不起", "没关系", "你好", "再见", "请",
    "帮助", "工作", "学习", "休息", "睡觉", "起床",
    "电话", "手机", "电脑", "电视", "电影", "音乐",
]

# 无手势检测标签
NO_GESTURE_LABEL = "<no_gesture>"
UNKNOWN_LABEL = "<unknown>"


# ======================================================================
# CSL 识别器（封装推理逻辑）
# ======================================================================

class CSLRecognizer:
    """【知识点：CSL 识别器 — 推理管线的封装】

    双模式设计（核心设计理念）：
      - 训练模式：用 Transformer 神经网络推理
      - 启发式模式：用几何规则（手指弯曲角度）推理
      权重存在 → 自动用神经网络；权重缺失 → 自动降级到启发式

    工作流程:
      1. 接收 MediaPipe 关键点（每帧 126 维或 63 维）
      2. 累积到滑动窗口缓冲区
      3. 缓冲区满 → 神经网络或启发式分类
      4. 稳定化过滤（连续 N 帧一致才确认）
      5. 冷却去重（同一手势短时间内不重复输出）
      6. 添加到 Token 列表

    【知识点：滑动窗口】
    不是每帧都分类，而是维护一个固定大小的缓冲区，满了才推理。
    窗口滑动时丢弃最旧的帧、加入最新的帧，这样保持持续识别。
    """

    # 【知识点：序列长度】Transformer 需要固定长度输入，30 帧约 2-3 秒的手势
    SEQUENCE_LENGTH = 12
    HEURISTIC_MIN_FRAMES = 3  # 启发式模式仅需少量帧

    def __init__(
        self,
        model_path: str | Path | None = None,
        num_classes: int = 100,
        confidence_threshold: float = 0.4,
        stability_threshold: int = 2,
        cooldown_frames: int = 15,
        device: str = "cpu",
        use_vit: bool = False,
        vit_dim: int = 768,
    ) -> None:
        self._model_path = Path(model_path) if model_path else None
        self._num_classes = num_classes
        self._confidence_threshold = confidence_threshold
        self._stability_threshold = stability_threshold
        self._cooldown_frames = cooldown_frames
        self._device = device
        self._use_vit = use_vit
        self._vit_dim = vit_dim

        # 输入维度: 双手关键点 (126=左手63+右手63) + 可选 ViT 特征 (768)
        self._base_dim = CSL_INPUT_DIM
        self._input_dim = self._base_dim + vit_dim if use_vit else self._base_dim

        self._model: nn.Module | None = None
        self._loaded = False
        self._vocabulary: list[str] = DEFAULT_CSL_VOCABULARY[:num_classes]

        # 序列缓冲区: 存储 (input_dim,) 特征向量
        self._sequence_buffer: list[np.ndarray] = []

        # ---- 稳定化状态 ----
        self._last_label: int | None = None
        self._stability_count: int = 0

        # 当前正在猜测的手势（实时更新，不确认会被后续手势替换）
        self._current_guess: str | None = None
        # 同一猜测持续帧数（用于自动确认）
        self._guess_stable_count: int = 0
        # 自动确认后冷却帧数（防止立即触发下一个）
        self._auto_cooldown: int = 0

        # 已确认的 Token 列表
        self._tokens: list[str] = []
        # 同词抑制：记录最后确认的词，短时间内不重复
        self._last_confirmed_token: str = ""
        self._same_token_cooldown: int = 0

        self._frame_count: int = 0

    # 自动确认：同一猜测持续 3 帧（~0.25s）不动 → 自动锁定
    AUTO_CONFIRM_FRAMES: int = 3
    # 自动确认后冷却 5 帧（~0.4s），给手切换时间
    AUTO_COOLDOWN_FRAMES: int = 5

    # ------------------------------------------------------------------
    # 词汇表管理
    # ------------------------------------------------------------------

    @property
    def vocabulary(self) -> list[str]:
        return self._vocabulary

    def set_vocabulary(self, vocab: list[str]) -> None:
        """设置词汇表（训练后替换默认词汇表）。"""
        self._vocabulary = vocab
        self._num_classes = len(vocab)

    def _load_vocabulary_from_file(self) -> None:
        """从训练时保存的词汇表文件加载自定义词汇表。"""
        from src.config import CSL_VOCABULARY_PATH

        if not CSL_VOCABULARY_PATH.exists():
            return

        try:
            import json
            with open(CSL_VOCABULARY_PATH, "r", encoding="utf-8") as f:
                vocab_dict: dict[str, int] = json.load(f)

            vocab = [""] * len(vocab_dict)
            for name, idx in vocab_dict.items():
                if 0 <= idx < len(vocab):
                    vocab[idx] = name

            self._vocabulary = vocab
            self._num_classes = len(vocab)
            logger.info(
                "已加载训练词汇表: %s (%d 个手势)",
                CSL_VOCABULARY_PATH, len(vocab),
            )
        except Exception:
            logger.warning("词汇表加载失败，使用默认词汇表", exc_info=True)

    def label_to_token(self, label_id: int) -> str:
        """将类别 ID 转换为 Token 文字。"""
        if 0 <= label_id < len(self._vocabulary):
            return self._vocabulary[label_id]
        return UNKNOWN_LABEL

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def load(self, model_path: str | Path | None = None) -> None:
        """
        加载训练好的模型权重。

        Args:
            model_path: .pt 或 .pth 权重文件路径。
                        如果为 None，使用初始化时指定的路径。
        """
        if model_path:
            self._model_path = Path(model_path)

        if self._loaded:
            return

        if not _TORCH_AVAILABLE:
            logger.warning("PyTorch 不可用，CSL 识别器将使用启发式模式")
            self._model = None  # 确保启发式触发
            self._loaded = True
            return

        # 检查权重文件是否存在
        weight_exists = (
            self._model_path is not None
            and self._model_path.exists()
        )

        if weight_exists:
            # 先加载 state_dict 来推断模型架构
            state_dict = torch.load(
                str(self._model_path),
                map_location=self._device,
                weights_only=True,
            )

            # 从 state_dict 推断模型架构参数
            d_model = state_dict["input_proj.weight"].shape[0]
            input_dim = state_dict["input_proj.weight"].shape[1]
            max_len = state_dict["pos_encoding.pe"].shape[1]

            # 统计层数
            num_layers = 0
            while f"encoder.layers.{num_layers}.norm1.weight" in state_dict:
                num_layers += 1

            # 推断 nhead: in_proj_weight shape = (3*d_model, d_model)
            # nhead 必须能被 d_model 整除
            in_proj_0 = state_dict["encoder.layers.0.self_attn.in_proj_weight"]
            # 找出所有可能的 nhead
            for nhead in [8, 4, 2, 1]:
                if d_model % nhead == 0:
                    break

            dim_feedforward = state_dict["encoder.layers.0.linear1.weight"].shape[0]
            actual_num_classes = state_dict["classifier.7.weight"].shape[0]

            self._input_dim = input_dim  # 更新为实际输入维度

            logger.info(
                "从权重推断架构: d_model=%d, layers=%d, nhead=%d, ff=%d, classes=%d, input=%d",
                d_model, num_layers, nhead, dim_feedforward, actual_num_classes, input_dim,
            )

            self._model = CSLTransformer(
                num_classes=actual_num_classes,
                input_dim=input_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
            )
            logger.info("加载 CSL 模型权重: %s", self._model_path)
            self._model.load_state_dict(state_dict)
            self._model.to(self._device)
            self._model.eval()
            self._num_classes = actual_num_classes
            logger.info(
                "CSL 模型加载完成 (%d 类, %s, 权重路径: %s)",
                actual_num_classes,
                self._device,
                self._model_path,
            )
            # 加载训练词汇表
            self._load_vocabulary_from_file()
        else:
            # ★ 关键修复: 权重缺失时不创建随机模型，
            # 设 _model = None 让 _infer() 进入启发式模式
            self._model = None
            logger.info(
                "CSL 模型权重未找到: %s\n"
                "识别器将使用启发式规则（手指伸展模式 → 数字手势分类）。\n"
                "要使用训练模型，请放入权重文件到: models/sign_language_model/pretrained/csl_model.pt",
                self._model_path or "models/sign_language_model/pretrained/csl_model.pt",
            )

        self._loaded = True

    def unload(self) -> None:
        """释放模型资源。"""
        self._model = None
        self._loaded = False
        self.clear()

    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_type(self) -> str:
        """返回当前识别模式描述。"""
        mode = "Transformer Encoder (训练权重)" if self._model is not None else "启发式规则 (距离比值法数字分类)"
        if self._use_vit:
            mode += " + ViT-B/16 视觉特征"
        return mode

    @property
    def has_trained_weights(self) -> bool:
        """是否加载了训练好的神经网络权重。"""
        return self._model is not None

    def get_diagnostic_info(self) -> dict:
        """返回诊断信息字典。"""
        return {
            "model_type": self.model_type,
            "has_trained_weights": self.has_trained_weights,
            "loaded": self._loaded,
            "vocabulary_size": len(self._vocabulary),
            "vocabulary_first_10": self._vocabulary[:10],
            "sequence_buffer_frames": len(self._sequence_buffer),
            "frame_count": self._frame_count,
            "tokens_collected": len(self._tokens),
            "tokens": list(self._tokens),
            "min_frames_required": (
                self.HEURISTIC_MIN_FRAMES if self._model is None
                else self.SEQUENCE_LENGTH // 2
            ),
        }

    # ------------------------------------------------------------------
    # 逐帧分类
    # ------------------------------------------------------------------

    def classify_frame(
        self,
        landmarks: np.ndarray,
        confidence_hint: float = 0.5,
        vit_features: np.ndarray | None = None,
    ) -> str | None:
        """
        对单帧关键点进行分类。

        Args:
            landmarks: (63,) 扁平化关键点或多帧 (T, 63)
            confidence_hint: MediaPipe 检测置信度（用于稳定性判断）
            vit_features: 可选，(768,) ViT-B/16 视觉特征向量，启用时与关键点拼接

        Returns:
            识别到的 Token 字符串，无稳定结果时返回 None
        """
        self._frame_count += 1

        if self._use_vit and vit_features is not None:
            feature = np.concatenate([landmarks, vit_features])
        else:
            feature = landmarks

        # 将特征加入序列缓冲区
        if feature.ndim == 2:
            for f in feature:
                self._sequence_buffer.append(f)
        else:
            self._sequence_buffer.append(feature)

        # 限制缓冲区大小
        while len(self._sequence_buffer) > self.SEQUENCE_LENGTH:
            self._sequence_buffer.pop(0)

        # ★ 识别启动：少量帧即可开始猜测，确认需要持续稳定性
        min_frames = (
            self.HEURISTIC_MIN_FRAMES if self._model is None
            else max(4, self.SEQUENCE_LENGTH // 4)  # 4帧即可开始猜测
        )

        if len(self._sequence_buffer) < min_frames:
            if self._frame_count % 10 == 0:
                logger.debug(
                    "缓冲中: %d/%d 帧 (模式: %s)",
                    len(self._sequence_buffer),
                    min_frames,
                    "启发式" if self._model is None else "Transformer",
                )
            return None

        # 使用模型推理
        label_id, confidence = self._infer()

        if label_id is None:
            return None

        # 过滤无手势
        if label_id >= len(self._vocabulary):
            self._stability_count = max(0, self._stability_count - 1)
            return None

        # 稳定化：同一标签连续出现 N 次才确认
        if label_id == self._last_label:
            self._stability_count += 1
        else:
            self._last_label = label_id
            self._stability_count = 1

        if self._stability_count < self._stability_threshold:
            return None

        token = self.label_to_token(label_id)

        # 自动确认冷却中：不更新猜测，等待手势切换
        if self._auto_cooldown > 0:
            self._auto_cooldown -= 1
            return None

        # 持续更新当前猜测，追踪稳定程度
        old = self._current_guess
        self._current_guess = token

        if old == token:
            self._guess_stable_count += 1
        else:
            self._guess_stable_count = 1
            if old is not None:
                logger.info(
                    "🖐 猜测变化: '%s' → '%s' (class_id=%d, conf=%.3f)",
                    old, token, label_id, confidence,
                )

        # 同词抑制冷却
        if self._same_token_cooldown > 0:
            self._same_token_cooldown -= 1

        # 自动确认：同一猜测持续 N 帧 → 自动锁定
        if self._guess_stable_count >= self.AUTO_CONFIRM_FRAMES:
            # 同词抑制：同一个词在 3x 冷却期内不重复确认
            if token == self._last_confirmed_token and self._same_token_cooldown > 0:
                self._current_guess = None
                self._guess_stable_count = 0
                self._stability_count = 0
                self._last_label = None
                return None

            # 黑名单过滤：与上一个已确认词相邻不合逻辑则丢弃
            if self._tokens and (self._tokens[-1], token) in CSL_TOKEN_BLACKLIST:
                logger.info("🚫 黑名单过滤: '%s' → '%s'（不合理相邻，已丢弃）",
                           self._tokens[-1], token)
                self._current_guess = None
                self._guess_stable_count = 0
                self._stability_count = 0
                self._last_label = None
                self._auto_cooldown = self.AUTO_COOLDOWN_FRAMES
                return None

            self._tokens.append(token)
            self._last_confirmed_token = token
            self._same_token_cooldown = self.AUTO_COOLDOWN_FRAMES * 3
            logger.info("🔒 自动确认: '%s' (稳定 %d 帧, 累计 %d 个)",
                       token, self._guess_stable_count, len(self._tokens))
            self._current_guess = None
            self._guess_stable_count = 0
            self._stability_count = 0
            self._last_label = None
            self._auto_cooldown = self.AUTO_COOLDOWN_FRAMES

        return token

    def _infer(self) -> tuple[int | None, float]:
        """
        执行模型推理。

        Returns:
            (label_id, confidence) 或 (None, 0.0)
        """
        if self._model is None or not _TORCH_AVAILABLE:
            return self._heuristic_infer()

        try:
            seq = np.stack(self._sequence_buffer[-self.SEQUENCE_LENGTH:])
            # 如果不足 SEQUENCE_LENGTH，在前面补零
            if seq.shape[0] < self.SEQUENCE_LENGTH:
                pad = np.zeros(
                    (self.SEQUENCE_LENGTH - seq.shape[0], seq.shape[1]),
                    dtype=seq.dtype,
                )
                seq = np.concatenate([pad, seq], axis=0)

            x = torch.from_numpy(seq).unsqueeze(0).float().to(self._device)
            with torch.no_grad():
                logits = self._model(x)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            best_id = int(np.argmax(probs))
            best_conf = float(probs[best_id])

            if best_conf < self._confidence_threshold:
                return None, 0.0

            return best_id, best_conf
        except Exception:
            logger.debug("CSL 模型推理失败，降级为启发式", exc_info=True)
            return self._heuristic_infer()

    def _heuristic_infer(self) -> tuple[int | None, float]:
        """【知识点：启发式规则 — 几何特征分类】

        这是不依赖神经网络的手势分类方法。核心思想：
        用 MediaPipe 关键点的几何关系（指尖到手腕距离）判断手指是否伸展，
        然后用手指伸展模式（哪几根手指是直的）匹配到具体手势。

        算法步骤：
          1. 取最近一帧的关键点 (21, 3)
          2. 对每根手指算 ratio = dist(指尖→手腕) / dist(MCP→手腕)
          3. ratio > 1.35 → 手指伸展（指尖远离手腕）
             ratio < 1.35 → 手指弯曲（指尖靠近手腕）
          4. 5 根手指的伸展/弯曲状态形成一个 5 位布尔模式
          5. 查表匹配 → Token 输出

        【知识点：为什么不直接用角度？】
        2D 图像中手指角度受视角影响大。距离比值法用 3D 坐标，更鲁棒。

        手势映射表（5位布尔模式 → 手势名）：
          拇指,食指,中指,无名,小指 → 手势
          (True, False, False, False, False) → 点赞
          (False, True, False, False, False) → 你/我（食指指人/指自己）
          (False, True, True, False, False) → 胜利/2
          ...共 32 种组合，映射约 12 种手势
        """
        if len(self._sequence_buffer) < 3:
            return None, 0.0

        landmarks = self._sequence_buffer[-1]

        if landmarks.shape[0] < 63:
            return None, 0.0

        # 126维双手模式 → 取非零半手用于启发式分类
        if landmarks.shape[0] >= 126:
            left_half = landmarks[:63]
            right_half = landmarks[63:126]
            # 优先用右手，右手全零则用左手
            if np.abs(right_half).sum() > 1e-6:
                landmarks = right_half
            else:
                landmarks = left_half

        # 重塑为 (21, 3)
        lm = landmarks[:63].reshape(21, 3)

        # ---- 指关节索引 ----
        fingertip = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky tips
        finger_mcp = [2, 5, 9, 13, 17]   # MCP 关节
        wrist = lm[0]                      # 手腕

        # ==== 距离比值法判断手指伸展 ====
        # 核心公式: ratio = |指尖 - 手腕| / |MCP - 手腕|
        # 伸展的手指：ratio > 1.35（指尖明显比指根远）
        # 弯曲的手指：ratio ≈ 1.0（指尖和指根差不多远）
        finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        extended: dict[str, bool] = {}
        ratios: dict[str, float] = {}
        for i, name in enumerate(finger_names):
            mcp = lm[finger_mcp[i]]
            tip = lm[fingertip[i]]

            # 【知识点：欧几里得距离】3D空间中两点的直线距离
            dist_tip_wrist = float(np.linalg.norm(tip - wrist))
            dist_mcp_w = float(np.linalg.norm(mcp - wrist))

            if dist_mcp_w < 1e-6:  # 防止除零
                extended[name] = False
                ratios[name] = 1.0
                continue

            ratio = dist_tip_wrist / dist_mcp_w
            ratios[name] = ratio

            # 【知识点：阈值工程】不同手指运动范围不同，阈值需要单独调
            threshold = 1.25 if name == "thumb" else 1.35
            extended[name] = ratio > threshold

        # ==== 5位布尔模式 → 手势查表 ====
        # 【知识点：查表法】5 根手指共 2^5=32 种组合，用字典直接映射
        digit_pattern = (
            extended["thumb"],
            extended["index"],
            extended["middle"],
            extended["ring"],
            extended["pinky"],
        )

        DIGIT_PATTERN_MAP = {
            # ---- 常用手势 (按特异性排序) ----
            (True, False, False, False, False):  "点赞",       # 👍 纯拇指竖起
            (True, True, False, False, True):   "我爱你",     # 🤟 ILY手形 (拇指+食指+小指)
            (False, False, False, False, False): "讨厌",      # ✊ 握拳
            (True, False, False, False, True):  "打电话",     # 🤙 拇指+小指 (电话手形)
            (True, True, False, False, False):  "OK",         # 👌 拇指+食指 (OK手势)
            (True, False, True, False, True):   "摇滚",       # 🤘 拇指+中指+小指 (摇滚)
            (True, True, True, False, True):    "牛",         # 🤙 拇指+食指+中指+小指 (666/厉害)
            (False, True, False, False, True):  "手枪",       # 👆 食指+小指
            # ---- 数字/指向 ----
            (False, True, False, False, False): "你",         # ☝ 单食指
            (False, True, True, False, False):  "胜利",       # ✌ 食指+中指 (胜利/耶/2)
            (False, True, True, True, False):   "3",
            (False, True, True, True, True):    "4",
            (True, True, True, True, True):     "5",
        }

        token = DIGIT_PATTERN_MAP.get(digit_pattern)

        # 【知识点：上下文区分】同样的手指模式可以有不同含义，需结合位置判断
        # 例如"你"和"我"都是单食指伸出，区别在于手的位置：
        #   手在下半部 + 食指尖低于手腕 → 指向自己 → "我"
        #   手在上半部 + 食指尖高于手腕 → 指向对方 → "你"
        if token == "你" and len(self._sequence_buffer) >= 3:
            lm_reshaped = landmarks.reshape(21, 3)
            wrist_y = float(lm_reshaped[0][1])  # 手腕y坐标 (0=顶部, 1=底部)
            index_tip = lm_reshaped[8]           # 食指尖
            wrist_pos = lm_reshaped[0]
            # 手在画面中下部 + 食指尖位置低于手腕 → 指向自己
            if wrist_y > 0.55 and index_tip[1] > wrist_pos[1]:
                token = "我"

        # 日志输出（每5帧）
        if self._frame_count % 5 == 0:
            ratios_str = ", ".join(f"{n}={ratios.get(n, 0):.2f}" for n in finger_names)
            logger.info(
                "启发式: pattern=%s → token=%s [%s]",
                digit_pattern, token or "无匹配", ratios_str,
            )

        if token is not None and token in self._vocabulary:
            idx = self._vocabulary.index(token)
            # 置信度基于比值裕量
            avg_ratio = sum(ratios.values()) / len(ratios) if ratios else 0
            conf = 0.55 + min(0.35, (avg_ratio - 1.2) / 1.5)
            conf = float(np.clip(conf, 0.4, 0.9))
            return idx, conf

        return None, 0.0

    # ------------------------------------------------------------------
    # Token 管理
    # ------------------------------------------------------------------

    def get_tokens(self) -> list[str]:
        """获取已确认的 Token 列表。"""
        return list(self._tokens)

    def get_guess(self) -> str | None:
        """获取当前实时猜测，不确认则会被后续手势替换。"""
        return self._current_guess

    def confirm_current(self) -> str | None:
        """将当前猜测锁定到 Token 列表，清空猜测状态准备下一个手势。"""
        if self._current_guess is not None:
            token = self._current_guess
            # 黑名单过滤
            if self._tokens and (self._tokens[-1], token) in CSL_TOKEN_BLACKLIST:
                logger.info("🚫 黑名单过滤(手动): '%s' → '%s'（不合理相邻，已丢弃）",
                           self._tokens[-1], token)
                self._current_guess = None
                self._stability_count = 0
                self._last_label = None
                return None
            self._tokens.append(token)
            logger.info("🔒 确认 Token: '%s' (累计 %d 个)", token, len(self._tokens))
            confirmed = token
            self._current_guess = None
            self._stability_count = 0
            self._last_label = None
            return confirmed
        return None

    def delete_token(self, index: int) -> bool:
        """删除指定索引的已确认 Token。"""
        if 0 <= index < len(self._tokens):
            removed = self._tokens.pop(index)
            logger.info("🗑 删除 Token[%d]='%s'", index, removed)
            return True
        return False

    def compose_sentence(self) -> str:
        """将已确认的 Token 拼接为待翻译的词语列表。"""
        return " ".join(self._tokens)

    def token_info(self) -> dict:
        """返回当前状态：猜测 + 已确认 Token 列表。"""
        progress = min(1.0, self._guess_stable_count / self.AUTO_CONFIRM_FRAMES) if self._current_guess else 0
        return {
            "guess": self._current_guess,
            "tokens": [{"index": i, "text": t} for i, t in enumerate(self._tokens)],
            "auto_progress": progress,  # 0~1，前端可显示确认进度条
        }

    def clear(self) -> None:
        """清空所有状态。"""
        self._sequence_buffer.clear()
        self._last_label = None
        self._stability_count = 0
        self._current_guess = None
        self._guess_stable_count = 0
        self._auto_cooldown = 0
        self._tokens.clear()
        self._frame_count = 0

    # ------------------------------------------------------------------
    # 序列保存/加载（用于训练数据收集）
    # ------------------------------------------------------------------

    def save_landmark_sequence(self, path: str | Path) -> None:
        """保存当前关键点序列缓冲区到文件（用于训练数据收集）。"""
        if not self._sequence_buffer:
            return
        data = np.stack(self._sequence_buffer)
        np.save(str(path), data)
        logger.info("关键点序列已保存到 %s (%d 帧)", path, data.shape[0])

    def load_landmark_sequence(self, path: str | Path) -> np.ndarray:
        """从文件加载关键点序列。"""
        return np.load(str(path))
