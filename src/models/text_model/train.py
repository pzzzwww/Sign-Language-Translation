"""
Qwen2 连词成句 LoRA 微调脚本。

将手语识别 token 重组为通顺中文句子。
训练数据格式: {"input": "我 喜欢 你", "output": "我喜欢你。"}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# 项目路径
ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import TEXT_MODEL_NAME

# ===================== 配置 =====================
MODEL_NAME = TEXT_MODEL_NAME
TRAIN_FILE = Path(__file__).parent / "word2sent.json"
LORA_OUTPUT = Path(__file__).parent / "lora_word2sent"
MAX_LENGTH = 128
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3

# ===================== 设备检测 =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
use_fp16 = device == "cuda"
print(f"设备: {device} | fp16: {use_fp16}")

# ===================== 加载分词器 =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===================== 加载模型 =====================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if use_fp16 else torch.float32,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True,
)

# ===================== LoRA 配置 =====================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===================== 加载数据集 =====================
with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

_ = """你是词语重组助手。将输入的全部词语组成通顺自然的中文句子。
规则：必须使用全部词语、不增不减不换、可加标点和连接词、只输出句子。"""

_SYSTEM_PROMPT = (
    "你是词语重组助手。将输入的全部词语组成通顺自然的中文句子。"
    "规则：必须使用全部词语、不增不减不换、可加标点和连接词、只输出句子。"
)


def format_example(example: dict) -> dict:
    return {
        "text": (
            f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{example['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['output']}<|im_end|>"
        )
    }


dataset = Dataset.from_list(data).map(format_example)


def tokenize_func(examples: dict) -> dict:
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )


token_data = dataset.map(tokenize_func)

# ===================== 训练参数 =====================
train_args = TrainingArguments(
    output_dir=str(LORA_OUTPUT),
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    logging_steps=5,
    save_strategy="epoch",
    fp16=use_fp16,
    bf16=False,
    optim="adamw_torch",
    report_to="none",
    use_cpu=device == "cpu",
)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ===================== 训练 =====================
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=token_data,
    data_collator=collator,
)

trainer.train()

# 保存 LoRA 权重
model.save_pretrained(str(LORA_OUTPUT))
tokenizer.save_pretrained(str(LORA_OUTPUT))
print(f"训练完成！LoRA 保存在: {LORA_OUTPUT}")
