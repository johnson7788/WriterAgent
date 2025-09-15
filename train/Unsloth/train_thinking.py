#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/7 08:06
# @File  : train_thinking.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :
# @Desc  : Qwen3 Thinking 训练（参考官方 notebook），复用 unsloth_core

from __future__ import annotations
import time
import argparse
import json
import dotenv
import logging
from dataclasses import asdict
from datasets import load_dataset, Dataset
from unsloth_core import (
    clip_display_text,
    TrainConfig,
    setup_logging,
    set_seed,
    log_env_info,
    setup_wandb,
    wandb_on_error,
    wandb_on_success,
    build_model_and_tokenizer,
    build_trainer,
    train_and_report,
    save_model,
)
dotenv.load_dotenv()


def parse_bool_flag(parser: argparse.ArgumentParser, true_flag: str, false_flag: str, default: bool):
    """同时支持 --flag / --no_flag 的布尔开关。返回存入 args 的目标名。"""
    dest = true_flag.replace("--", "").replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(true_flag, dest=dest, action="store_true")
    group.add_argument(false_flag, dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})
    return dest


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Thinking SFT（unsloth_core 驱动）")

    # 与原脚本等价的关键默认值
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3-4B-Thinking-2507")
    parser.add_argument("--max_seq_length", type=int, default=TrainConfig.max_seq_length)
    parse_bool_flag(parser, "--load_in_4bit", "--no_load_in_4bit", default=TrainConfig.load_in_4bit)
    parse_bool_flag(parser, "--load_in_8bit", "--no_load_in_8bit", default=TrainConfig.load_in_8bit)
    parse_bool_flag(parser, "--full_finetuning", "--no_full_finetuning", default=TrainConfig.full_finetuning)
    parser.add_argument("--hf_token", type=str, default=None)

    parser.add_argument("--lora_r", type=int, default=TrainConfig.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=TrainConfig.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=TrainConfig.lora_dropout)

    # Thinking 专用 chat 模板
    parser.add_argument("--chat_template", type=str, default="qwen3-thinking")
    parser.add_argument("--dataset_name", type=str, default="unsloth/OpenMathReasoning-mini")
    parser.add_argument("--dataset_split", type=str, default="cot")

    parser.add_argument("--per_device_train_batch_size", type=int, default=TrainConfig.per_device_train_batch_size)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=TrainConfig.gradient_accumulation_steps)
    parser.add_argument("--warmup_steps", type=int, default=TrainConfig.warmup_steps)
    parser.add_argument("--max_steps", type=int, default=TrainConfig.max_steps)
    parser.add_argument("--num_train_epochs", type=float, default=TrainConfig.num_train_epochs)
    parser.add_argument("--learning_rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--logging_steps", type=int, default=TrainConfig.logging_steps)
    parser.add_argument("--optim", type=str, default=TrainConfig.optim)
    parser.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--lr_scheduler_type", type=str, default=TrainConfig.lr_scheduler_type)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--report_to", type=str, default=TrainConfig.report_to)

    parser.add_argument("--data_files", type=str, default=None,help="逗号分隔的本地数据文件，如 data/train.jsonl")

    parser.add_argument("--output_dir", type=str, default="./outputs/qwen3_4b_thinking_lora")
    parser.add_argument("--save_steps", type=int, default=TrainConfig.save_steps)
    parser.add_argument("--save_total_limit", type=int, default=None)

    # W&B 相关开关
    parse_bool_flag(parser, "--use_wandb", "--no_use_wandb", default=TrainConfig.use_wandb)
    parser.add_argument("--wandb_project", type=str, default="train_thinking")
    parser.add_argument("--wandb_entity", type=str, default=TrainConfig.wandb_entity)
    parser.add_argument("--wandb_run_name", type=str, default=TrainConfig.wandb_run_name)
    parser.add_argument("--wandb_group", type=str, default=TrainConfig.wandb_group)
    parser.add_argument("--wandb_job_type", type=str, default=TrainConfig.wandb_job_type)
    parser.add_argument("--wandb_mode", type=str, default=TrainConfig.wandb_mode)
    parser.add_argument("--wandb_dir", type=str, default=TrainConfig.wandb_dir)
    parser.add_argument("--wandb_notes", type=str, default=TrainConfig.wandb_notes)
    parser.add_argument("--wandb_log_model", type=str, default=str(TrainConfig.wandb_log_model))
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None, help="空格分隔的 tag 列表")

    args = parser.parse_args()

    # 解析 wandb_log_model 为 bool/str
    wandb_log_model: bool | str
    if str(args.wandb_log_model).lower() in {"true", "1", "yes"}:
        wandb_log_model = True
    elif str(args.wandb_log_model).lower() in {"false", "0", "no"}:
        wandb_log_model = False
    else:
        wandb_log_model = str(args.wandb_log_model)

    cfg = TrainConfig(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        full_finetuning=args.full_finetuning,
        hf_token=args.hf_token,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        chat_template=args.chat_template,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        report_to=args.report_to,
        data_files=args.data_files,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags or TrainConfig().wandb_tags,
        wandb_dir=args.wandb_dir,
        wandb_group=args.wandb_group,
        wandb_job_type=args.wandb_job_type,
        wandb_mode=args.wandb_mode,
        wandb_notes=args.wandb_notes,
        wandb_log_model=wandb_log_model,
    )
    # Thinking 模型同样使用 responses-only 的分隔符（与原始脚本一致）
    cfg.instruction_part = "<|im_start|>user\n"
    cfg.response_part = "<|im_start|>assistant\n"
    cfg.dataset_text_field = "text"
    return cfg

def _safe_json_dumps(obj) -> str:
    """尽量完整地打印样本，避免 numpy/int64 等类型导致的序列化报错"""
    def _default(o):
        try:
            import numpy as np
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
        except Exception:
            pass
        return str(o)
    return json.dumps(obj, ensure_ascii=False, indent=2, default=_default)


def prepare_dataset_openmath_thinking(cfg: TrainConfig, tokenizer, logger: logging.Logger, *, split: str = "cot") -> Dataset:
    """OpenMathReasoning-mini → conversations → text"""
    t0 = time.perf_counter()
    logger.info("加载数据集: %s [%s] …", cfg.dataset_name, cfg.dataset_split)
    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    logger.info("加载完成: %d 行，字段: %s", len(ds), ds.column_names)

    # 轻量校验
    for col in ("problem", "generated_solution"):
        if col not in ds.column_names:
            raise KeyError(f"缺少必需字段: {col}")

    # 简短预览（原始）
    if len(ds) > 0:
        logger.info("原始样本#0.problem: %s", clip_display_text(ds[0]["problem"]))
        logger.info("原始样本#0.solution: %s", clip_display_text(ds[0]["generated_solution"]))

    # 构造 conversations
    def _to_conversations(batch):
        convs = []
        for p, s in zip(batch["problem"], batch["generated_solution"]):
            if isinstance(p, str) and isinstance(s, str) and p.strip() and s.strip():
                convs.append([{"role": "user", "content": p.strip()},
                              {"role": "assistant", "content": s.strip()}])
        return {"conversations": convs}

    t1 = time.perf_counter()
    ds = ds.map(_to_conversations, batched=True, desc="build_conversations")
    logger.info("build_conversations 完成（用时 %.2fs）", time.perf_counter() - t1)

    # 预览（对话）
    if len(ds) > 0 and "conversations" in ds.column_names:
        c0 = ds[0]["conversations"]
        logger.info("对话样本#0.user: %s", clip_display_text(c0[0]["content"]))
        logger.info("对话样本#0.assistant: %s", clip_display_text(c0[1]["content"]))

    # 应用 chat 模板
    def _format(batch):
        texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
                 for c in batch["conversations"]]
        return {"text": texts}

    t2 = time.perf_counter()
    ds = ds.map(_format, batched=True, desc="apply_chat_template")
    logger.info("apply_chat_template 完成（用时 %.2fs）", time.perf_counter() - t2)

    # 预览（最终文本）
    if len(ds) > 0:
        if "text" in ds.column_names:
            logger.info("模板化样本#0.text(截断预览): %s", clip_display_text(ds[0]["text"]))
        try:
            logger.info("==== 样本#0 完整记录（OpenMath pipeline 之后） ====\n%s", _safe_json_dumps(ds[0]))
        except Exception as e:
            logger.warning("打印完整样本失败: %r", e)
    logger.info("数据集准备完成：%d 行，总耗时 %.2fs", len(ds), time.perf_counter() - t0)
    return ds

def _parse_data_files_arg(s: str) -> dict | list | str:
    """
    将 --data_files 的字符串解析为 datasets.load_dataset(data_files=...) 接受的形式。
    支持：
      - "data/train.jsonl"
      - "train=data/train-*.jsonl"
      - "train=... , validation=..."
      - "a.jsonl,b.jsonl"（等价于单一 'train' 拆分的多文件列表）
    """
    s = (s or "").strip()
    if not s:
        return s
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if all("=" in p for p in parts):
        out = {}
        for p in parts:
            k, v = p.split("=", 1)
            out[k.strip()] = v.strip()
        return out  # e.g. {"train":"...", "validation":"..."}
    if len(parts) == 1:
        return parts[0]        # 单文件
    return parts               # 多文件列表 → 单一拆分

def prepare_dataset_from_json_files(cfg: TrainConfig, tokenizer, logger: logging.Logger) -> Dataset:
    """
    通用自定义数据集加载：
      - 接受 conversations/messages/input+output 其中一种
      - 统一生成 'text' 列供 SFT 使用
    """
    assert cfg.data_files, "cfg.data_files 不能为空"
    df_arg = _parse_data_files_arg(cfg.data_files)

    logger.info("从本地文件加载数据: %s", df_arg)
    raw = load_dataset("json", data_files=df_arg)

    # 选择拆分：优先 cfg.dataset_split；否则取第一个可用键；若 load_dataset 返回 Dataset 则直接用
    if isinstance(raw, dict) or hasattr(raw, "keys"):  # DatasetDict
        keys = list(raw.keys())
        split = cfg.dataset_split if cfg.dataset_split in keys else (keys[0] if keys else "train")
        ds = raw[split]
        logger.info("已选择拆分: %s（可用: %s）", split, keys)
    else:
        ds = raw

    cols = set(ds.column_names)
    logger.info("原始字段: %s", cols)

    # 预处理为 conversations
    if "conversations" in cols:
        conv_field = "conversations"
    elif "messages" in cols:
        conv_field = "messages"
    elif {"input", "output"}.issubset(cols):
        conv_field = None
        def to_conv_io(batch):
            convs = []
            for x, y in zip(batch["input"], batch["output"]):
                if isinstance(x, str) and isinstance(y, str) and x.strip() and y.strip():
                    convs.append([{"role": "user", "content": x.strip()},
                                  {"role": "assistant", "content": y.strip()}])
            return {"conversations": convs}
        ds = ds.map(to_conv_io, batched=True, desc="build_conversations_from_io")
        conv_field = "conversations"
    elif "text" in cols:
        # 已经是最终文本，直接返回
        logger.info("检测到已有 'text' 字段，跳过模板化。")
        return ds
    else:
        raise KeyError("未检测到支持的数据格式。需要 'conversations'/'messages' 或 'input'+'output'，"
                       "或直接提供 'text'。")

    # 应用 chat 模板 → 生成 text
    def _format(batch):
        texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
                 for c in batch[conv_field]]
        return {"text": texts}
    ds = ds.map(_format, batched=True, desc="apply_chat_template")
    if len(ds) > 0:
        try:
            logger.info("==== 样本#0 完整记录（自定义数据 pipeline 之后） ====\n%s",_safe_json_dumps(ds[0]))
        except Exception as e:
            logger.warning("打印完整样本失败: %r", e)
    logger.info("自定义数据集准备完成：%d 行。", len(ds))
    return ds

def main(cfg: TrainConfig | None = None) -> None:
    cfg = cfg or TrainConfig()

    # 日志
    logger = setup_logging(cfg.output_dir, cfg.logging_dir)

    # 打印配置
    logger.info("===== 训练配置 =====")
    logger.info(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    # 随机种子 & 环境信息
    set_seed(cfg.seed, logger)
    log_env_info(logger)

    # 初始化 W&B（尽早建立 run，记录环境/配置）
    run = setup_wandb(cfg, logger)

    # 构建模型与 tokenizer
    model, tokenizer = build_model_and_tokenizer(cfg, logger)

    # 准备数据集：优先使用本地 data_files；否则用 OpenMathReasoning

    if cfg.data_files:
        dataset = prepare_dataset_from_json_files(cfg, tokenizer, logger)
    else:
        dataset = prepare_dataset_openmath_thinking(cfg, tokenizer, logger)
    # 构建 Trainer
    trainer = build_trainer(model, tokenizer, dataset, cfg, logger)

    # 训练并报告
    stats = train_and_report(trainer, logger)

    # 保存模型 & 可选上传 artifact（别名：final）
    save_model(
        trainer,
        tokenizer,
        cfg.output_dir,
        logger,
        log_artifact=(cfg.wandb_log_model if cfg.wandb_log_model else False),
    )

    # 成功收尾
    extra = {"metrics/train_runtime_sec": float(stats.metrics.get("train_runtime", 0.0))}
    wandb_on_success(extra_summary=extra, exit_code=0)

    logger.info(f"{float(stats.metrics.get('train_runtime', 0.0)):.2f} 秒 used for training.")
    logger.info("🎉  训练结束。")

if __name__ == "__main__":
    main(parse_args())
