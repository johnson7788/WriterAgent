#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/7 08:04
# @File  : unsloth_core.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :
"""
Unsloth 训练公共模块
- 日志、W&B、种子/环境
- 模型 & Tokenizer 构建（LoRA/PEFT 注入）
- 数据准备（通用 conversations & Thinking/OpenMathReasoning）
- SFTTrainer 构建（responses only）
- 训练度量与保存

依赖：unsloth, trl, datasets, torch, numpy, (可选) wandb
"""
from __future__ import annotations
import os
import sys
import json
import time
import random
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

# 可选：W&B
try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
    print(f"WANDB_AVAILABLE是可用的，已经安装了wandb")
except Exception:
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False
    print(f"WANDB_AVAILABLE不可用，没有安装wandb")

# Unsloth / TRL / Datasets
from unsloth import FastModel
from unsloth.chat_templates import (
    get_chat_template,
    standardize_data_formats,
    train_on_responses_only,
)
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig


# ==============================
# 配置（可被 2 个主程序复用）
# ==============================
@dataclass
class TrainConfig:
    # 模型 / Tokenizer
    model_name: str = "unsloth/Qwen3-4B-Instruct-2507"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False
    hf_token: Optional[str] = None  # 若模型为 gated，可在此传递 token

    # LoRA / PEFT 设置
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    bias: str = "none"
    use_gradient_checkpointing: str | bool = "unsloth"  # True / False / "unsloth"
    use_rslora: bool = False
    loftq_config: Optional[dict] = None
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Chat 模板/数据处理
    chat_template: str = "qwen3-instruct"
    dataset_name: str = "mlabonne/FineTome-100k"
    dataset_split: str = "train"
    dataset_text_field: str = "text"  # 生成后的文本字段名

    # 仅训练 assistant 响应的分隔符（依据模板）：
    instruction_part: str = "<|im_start|>user\n"
    response_part: str = "<|im_start|>assistant\n"

    # 训练超参（传入 TRL.SFTConfig）
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 60  # 或使用 num_train_epochs
    num_train_epochs: Optional[int] = 2
    learning_rate: float = 2e-4
    logging_steps: int = 1
    optim: str = "adamw_8bit"  # 需 bitsandbytes
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    report_to: str = "wandb"  # 改为默认同步到 wandb
    data_files: Optional[str] = None  #自定义的训练的数据集

    # W&B 相关
    use_wandb: bool = True
    wandb_project: Optional[str] = "unsloth-sft"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    wandb_dir: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_job_type: str = "train"
    wandb_mode: Optional[str] = None  # None/"online"/"offline"/"disabled"
    wandb_notes: Optional[str] = None
    wandb_log_model: bool | str = False  # True/False/"checkpoint"/"end"

    # 输出/保存
    output_dir: str = "./outputs/qwen3_4b_sft_lora"
    save_steps: Optional[int] = 2  # 定期保存
    save_total_limit: Optional[int] = None
    logging_dir: Optional[str] = None  # 自定义日志目录


# ==============================
# 日志工具
# ==============================

def setup_logging(output_dir: str, logging_dir: Optional[str] = None) -> logging.Logger:
    """配置日志：控制台 + 文件。"""
    os.makedirs(output_dir, exist_ok=True)
    log_dir = logging_dir or os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, time.strftime("train_%Y%m%d_%H%M%S.log"))

    logger = logging.getLogger("unsloth_core")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # 控制台输出
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
    ch.setFormatter(ch_formatter)

    # 文件输出
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s:%(lineno)d - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fh_formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f"日志写入: {log_file}")
    return logger


# ==============================
# W&B 工具
# ==============================

def setup_wandb(cfg: TrainConfig, logger: logging.Logger):
    """初始化 wandb。返回 run 对象或 None。"""
    if not cfg.use_wandb:
        logger.info("已禁用 W&B。")
        return None
    if not WANDB_AVAILABLE:
        logger.warning("未检测到 wandb 包，已跳过 W&B 集成。pip install wandb")
        return None

    # 允许通过环境变量覆盖
    project = cfg.wandb_project or os.getenv("WANDB_PROJECT") or "unsloth-sft"
    entity = cfg.wandb_entity or os.getenv("WANDB_ENTITY")

    run = wandb.init(
        project=project,
        entity=entity,
        name=cfg.wandb_run_name,
        group=cfg.wandb_group,
        job_type=cfg.wandb_job_type,
        dir=cfg.wandb_dir,
        tags=cfg.wandb_tags or None,
        notes=cfg.wandb_notes,
        mode=cfg.wandb_mode,  # None 使用默认
        config=asdict(cfg),
        reinit=False,
    )

    # 记录当前脚本代码，便于复现
    try:
        wandb.run.log_code(root=str(Path(__file__).resolve().parent))
    except Exception:
        pass

    logger.info(f"已连接 W&B：project={project}, run={run.name}")
    return run


def wandb_log_metrics(metrics: dict):
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(metrics)


def wandb_on_error(e: Exception, logger: logging.Logger):
    if WANDB_AVAILABLE and wandb.run is not None:
        try:
            # 尝试发出告警（企业/团队版更友好），失败则降级为普通日志
            try:
                wandb.alert(title="Training crashed", text=str(e), level=wandb.AlertLevel.ERROR)
            except Exception:
                pass
            wandb.run.summary["status"] = "failed"
            wandb.log({"error/exception": str(e)})
            wandb.finish(exit_code=1)
            logger.error("已将异常上报至 W&B（status=failed）")
        except Exception as ee:
            logger.error(f"上报 W&B 异常失败: {ee}")


def wandb_on_success(extra_summary: dict | None = None, exit_code: int = 0):
    if WANDB_AVAILABLE and wandb.run is not None:
        if extra_summary:
            for k, v in extra_summary.items():
                wandb.run.summary[k] = v
        wandb.run.summary["status"] = "success"
        wandb.finish(exit_code=exit_code)


# ==============================
# 通用工具
# ==============================

def set_seed(seed: int, logger: logging.Logger | None = None) -> None:
    """设置随机种子，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # 更快训练；如需完全确定性可设 True
    torch.backends.cudnn.benchmark = True
    if logger:
        logger.info(f"随机种子已设置: {seed}")


def log_env_info(logger: logging.Logger) -> None:
    """记录环境与 GPU 信息。"""
    logger.info(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        max_mem = round(gpu.total_memory / 1024 / 1024 / 1024, 3)
        start_reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        logger.info(f"GPU: {gpu.name}  显存上限: {max_mem} GB  启动保留: {start_reserved} GB")
    else:
        logger.warning("未检测到可用 CUDA，训练将使用 CPU（可能非常慢）")


# ==============================
# 构建模型 & Tokenizer
# ==============================

def build_model_and_tokenizer(cfg: TrainConfig, logger: logging.Logger):
    """加载基础模型，并注入 LoRA 适配器；应用 Chat 模板。"""
    if cfg.load_in_4bit and cfg.load_in_8bit:
        logger.warning("4bit 与 8bit 仅能二选一，已优先使用 4bit，并关闭 8bit！")
        cfg.load_in_8bit = False

    logger.info("开始加载基础模型…")
    model, tokenizer = FastModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        load_in_8bit=cfg.load_in_8bit,
        full_finetuning=cfg.full_finetuning,
        token=cfg.hf_token,
    )
    logger.info("基础模型加载完成。")

    logger.info("注入 LoRA 适配器…")
    model = FastModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=cfg.target_modules,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.bias,
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        random_state=cfg.seed,
        use_rslora=cfg.use_rslora,
        loftq_config=cfg.loftq_config,
    )
    logger.info("LoRA 适配器注入完成。")

    logger.info(f"应用 Chat 模板: {cfg.chat_template}")
    tokenizer = get_chat_template(tokenizer, chat_template=cfg.chat_template)
    return model, tokenizer


# ==============================
# 数据集准备
# ==============================

def clip_display_text(s: str, limit: int = 4000) -> str:
    if s is None:
        return "None"
    s = str(s)
    return s if len(s) <= limit else (s[:limit] + f" … <+{len(s) - limit} chars>")

def display_json(obj, limit: int = 500) -> str:
    try:
        txt = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        txt = str(obj)
    return clip_display_text(txt, limit)

# ==============================
# Trainer / 训练 & 保存
# ==============================

def build_trainer(model, tokenizer, dataset: Dataset, cfg: TrainConfig, logger: logging.Logger) -> SFTTrainer:
    """创建 SFTTrainer，并设置仅训练回答段落。"""
    logger.info("创建 SFTTrainer…")

    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        dataset_text_field=cfg.dataset_text_field,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        max_steps=cfg.max_steps,
        # 也可选择设定 num_train_epochs（两者不要同时强制）
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        optim=cfg.optim,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        seed=cfg.seed,
        report_to=cfg.report_to,  # → "wandb" 时将自动上报
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=training_args,
    )

    logger.info("切换为仅学习 assistant 响应（忽略 user 指令部分的 loss）…")
    trainer = train_on_responses_only(
        trainer,
        instruction_part=cfg.instruction_part,
        response_part=cfg.response_part,
    )

    # 记录一个编码后的样本，确认 mask 是否合理（只做日志预览，不影响训练）
    try:
        decoded = tokenizer.decode(trainer.train_dataset[0]["input_ids"][:256])
        logger.info(f"编码样本预览: {decoded}")
    except Exception as e:
        logger.warning(f"编码样本预览失败: {e}")

    # 将梯度/权重变化发送到 W&B（可选）
    if WANDB_AVAILABLE and wandb.run is not None and cfg.use_wandb:
        try:
            wandb.watch(trainer.model, log="gradients", log_freq=max(1, cfg.logging_steps))
        except Exception:
            pass

    return trainer


# ==============================
# 训练与度量
# ==============================

def train_and_report(trainer: SFTTrainer, logger: logging.Logger):
    """启动训练并记录显存与耗时。"""
    gpu_stats = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    start_reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3) if torch.cuda.is_available() else 0.0
    max_mem = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3) if gpu_stats else 0.0

    if gpu_stats:
        logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_mem} GB.")
        logger.info(f"启动时保留显存 = {start_reserved} GB.")
        wandb_log_metrics({
            "env/gpu_name": gpu_stats.name,
            "env/gpu_mem_gb": max_mem,
            "memory/start_reserved_gb": start_reserved,
        })

    logger.info("开始训练…")
    trainer_stats = trainer.train()
    logger.info("训练完成。")

    used_reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3) if torch.cuda.is_available() else 0.0
    used_for_lora = round(used_reserved - start_reserved, 3)
    used_pct = round(used_reserved / max_mem * 100, 3) if max_mem else 0.0
    lora_pct = round(used_for_lora / max_mem * 100, 3) if max_mem else 0.0

    rt = float(trainer_stats.metrics.get('train_runtime', 0.0))
    logger.info(f"训练耗时 {rt:.2f} 秒（约 {rt/60:.2f} 分钟）。")
    logger.info(f"峰值保留显存 = {used_reserved} GB；其中训练增量 = {used_for_lora} GB。")
    if max_mem:
        logger.info(f"显存占用峰值占比 = {used_pct}%；训练增量占比 = {lora_pct}%。")

    # 自定义指标上报到 W&B
    wandb_log_metrics({
        "memory/peak_reserved_gb": used_reserved,
        "memory/peak_reserved_pct": used_pct,
        "memory/lora_delta_gb": used_for_lora,
        "memory/lora_delta_pct": lora_pct,
        "time/train_runtime_sec": rt,
        "trainer/global_step": getattr(trainer.state, "global_step", 0),
    })

    return trainer_stats


# ==============================
# 保存模型
# ==============================

def save_model(trainer: SFTTrainer, tokenizer, output_dir: str, logger: logging.Logger, *, log_artifact: bool | str = False) -> None:
    """尝试使用TRL/Transformers 保存，并可选上传到 W&B Artifact。"""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Unsloth 保存模型 Trainer.save_model")
    try:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"模型已保存至: {output_dir}")

        # 可选：上传模型目录为 artifact
        if log_artifact and WANDB_AVAILABLE and wandb.run is not None:
            art = wandb.Artifact(
                name=f"{Path(output_dir).name}-{wandb.run.id}",
                type="model",
                metadata={"framework": "transformers", "task": "sft"},
            )
            art.add_dir(output_dir)
            aliases = ["latest"]
            if isinstance(log_artifact, str):
                aliases.append(log_artifact)
            wandb.log_artifact(art, aliases=aliases)
            logger.info("模型已作为 W&B Artifact 上传。")
    except Exception as ee:
        logger.error(f"保存失败: {ee}")
        raise