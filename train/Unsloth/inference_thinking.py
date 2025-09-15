#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/7 09:20
# @File  : inference_thinking.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 基于 inference_sft.py 的推理脚本，适配 Qwen3-* Thinking 模型
#          - 自动识别 LoRA/完整模型与最新 checkpoint
#          - 复用 unsloth_core 配置与 chat template（qwen3-thinking）
#          - 可选隐藏/剥离 <think> 思维段落，仅输出最终回答
"""
用法示例

1）使用最终输出目录（已保存 tokenizer）

```bash
python inference_thinking.py \
  --ckpt_dir ./outputs/qwen3_4b_thinking_lora \
  --prompt "一道经典鸡兔同笼题：鸡和兔共有 20 个头，50 条腿，各有多少只？" \
  --max_new_tokens 256
```

2）直接指定某个 checkpoint（脚本也会自动挑最新）

```bash
python inference_thinking.py \
  --ckpt_dir ./outputs/qwen3_4b_thinking_lora/checkpoint-120 \
  --prompt "Solve: (x-3)^2 = 25"
```

3）当 `--ckpt_dir` 是 LoRA 适配器目录时（需要 `peft`）

```bash
python inference_thinking.py \
  --ckpt_dir ./outputs/qwen3_4b_thinking_lora \
  --base_model unsloth/Qwen3-4B-Thinking-2507 \
  --prompt "Explain breadth-first search."
```

4）合并 LoRA，得到纯模型权重（更快推理）

```bash
python inference_thinking.py \
  --ckpt_dir ./outputs/qwen3_4b_thinking_lora \
  --base_model unsloth/Qwen3-4B-Thinking-2507 \
  --merge_lora \
  --prompt "请给一道中考几何压轴题并详细解答。"
```

5）隐藏思维过程，仅打印最终回答（非流式）

```bash
python inference_thinking.py \
  --ckpt_dir ./outputs/qwen3_4b_thinking_lora \
  --prompt "2025 年 3 月共有多少个星期日？" \
  --strip_thought --no_stream
```
"""

from __future__ import annotations
import os
import re
import json
import argparse
import logging
from typing import List, Optional, Tuple

import torch
from transformers import TextStreamer, AutoTokenizer

# 复用训练公共模块
from unsloth_core import (
    TrainConfig,         # 直接复用默认模型名、模板、量化等默认值
    setup_logging,
    set_seed,
    log_env_info,
)
from unsloth.chat_templates import get_chat_template

# 可选：使用 Unsloth 加速加载
from unsloth import FastModel as _FastModel
from peft import PeftModel


# ==============================
# 文件/目录工具
# ==============================

def _file_exists(d: str, name: str) -> bool:
    return os.path.isfile(os.path.join(d, name))


def _detect_ckpt_type(path: str) -> str:
    """
    返回:
      - "adapter": 目录中包含 PEFT/LoRA 适配器
      - "full"   : 目录是完整模型（有 config.json 和模型权重）
      - "unknown": 未识别
    """
    if _file_exists(path, "adapter_config.json") or _file_exists(path, "adapter_model.safetensors"):
        return "adapter"
    if _file_exists(path, "config.json"):
        # 非严格判断：出现下列任意文件名即可视作完整模型
        for fname in ["model.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json", "model.safetensors.index.json"]:
            if _file_exists(path, fname):
                return "full"
    return "unknown"


def _find_latest_checkpoint(root: str) -> Optional[str]:
    """
    在训练输出根目录下挑选最新的 checkpoint-<global_step> 子目录。
    若不存在，返回 None。
    """
    if not os.path.isdir(root):
        return None
    subdirs = [d for d in os.listdir(root) if d.startswith("checkpoint-")]
    if not subdirs:
        return None

    def _step(dname: str) -> int:
        m = re.search(r"checkpoint-(\d+)", dname)
        return int(m.group(1)) if m else -1

    subdirs.sort(key=_step, reverse=True)
    for d in subdirs:
        path = os.path.join(root, d)
        if os.path.isdir(path) and _detect_ckpt_type(path) in {"adapter", "full"}:
            return path
    return None


# ==============================
# 参数解析
# ==============================

def parse_args(cfg_defaults: TrainConfig) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unsloth Thinking 推理脚本（复用训练公共模块）")

    # 权重与设备
    p.add_argument("--ckpt_dir", type=str, default="outputs/qwen3_4b_thinking_lora",
                   help="训练产物目录（可为最终输出目录或 checkpoint-* 子目录）。")
    p.add_argument("--base_model", type=str, default="unsloth/Qwen3-4B-Thinking-2507",
                   help="当 --ckpt_dir 为 LoRA 适配器目录时，用于加载的基础模型名/路径。")
    p.add_argument("--max_seq_length", type=int, default=cfg_defaults.max_seq_length)
    p.add_argument("--load_in_4bit", action="store_true", default=cfg_defaults.load_in_4bit)
    p.add_argument("--no_load_in_4bit", action="store_true", help="关闭 4bit（优先级高于 --load_in_4bit）")
    p.add_argument("--load_in_8bit", action="store_true", default=cfg_defaults.load_in_8bit)
    p.add_argument("--no_load_in_8bit", action="store_true", help="关闭 8bit（优先级高于 --load_in_8bit）")
    p.add_argument("--merge_lora", action="store_true", help="加载 LoRA 后将权重合并并卸载适配器（减少推理开销）")

    # 模板/随机种子
    p.add_argument("--chat_template", type=str, default="qwen3-thinking")
    p.add_argument("--seed", type=int, default=cfg_defaults.seed)

    # 生成参数
    p.add_argument("--system", type=str, default="你是小森智能体（XiaoSen Health Agent）", help="系统prompt")
    p.add_argument("--prompt", type=str, default=None, help="单条 user 提示词。若提供则覆盖默认示例")
    p.add_argument("--messages_json", type=str, default=None,
                   help="包含 messages(list[{'role','content'}]) 的 JSON 文件路径。")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--no_stream", action="store_true", help="关闭流式输出")

    # Thinking 专用输出控制
    p.add_argument("--strip_thought", action="store_true",
                   help="剥离思维过程（<think>/思考段落），仅打印最终回答。会强制使用非流式模式。")

    args = p.parse_args()

    # 布尔修正
    if args.no_load_in_4bit:
        args.load_in_4bit = False
    if args.no_load_in_8bit:
        args.load_in_8bit = False
    # 仅允许二选一
    if args.load_in_4bit and args.load_in_8bit:
        args.load_in_8bit = False  # 与训练一致，4bit 优先

    # 非流式剥离思维
    if args.strip_thought and not args.no_stream:
        args.no_stream = True  # 需要后处理文本

    return args


# ==============================
# 数据准备
# ==============================

def _load_messages(args: argparse.Namespace) -> List[dict]:
    if args.messages_json:
        with open(args.messages_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        msgs = data["messages"] if isinstance(data, dict) and "messages" in data else data
        assert isinstance(msgs, list), "messages_json 必须包含一个 list"
        return msgs
    if args.prompt:
        if args.system:
            return [
                    {"role": "system", "content": args.system.strip()},
                    {"role": "user", "content": args.prompt.strip()},
                ]
        else:
            return [{"role": "user", "content": args.prompt.strip()}]
    # 默认示例：鼓励使用思维链
    return [
        {"role": "user", "content": "解方程：2x + 3 = 15。请给出推理过程与最终答案。"}
    ]


# ==============================
# 加载 tokenizer / model
# ==============================

def _load_tokenizer_from_dir_or_base(ckpt_dir: str, base_model: str, chat_template: str, logger: logging.Logger):
    """
    优先从 ckpt_dir 读取 tokenizer（训练已保存 tokenizer.save_pretrained），否则退回 base_model。
    统一应用 chat_template。
    """
    tok_dir = ckpt_dir if _file_exists(ckpt_dir, "tokenizer_config.json") or _file_exists(ckpt_dir, "tokenizer.json") else base_model
    logger.info(f"加载 Tokenizer 自: {tok_dir}")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
    return tokenizer


def _load_full_model(ckpt_dir: str, args: argparse.Namespace, logger: logging.Logger):
    """
    加载完整模型（非 LoRA 适配器）。优先使用 Unsloth 的 FastModel（若可用）。
    """
    logger.info("检测到完整模型目录，开始加载…")
    model, _ = _FastModel.from_pretrained(
        model_name=ckpt_dir,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        full_finetuning=False,
        token=None,
    )
    return model


def _load_adapter_model(ckpt_dir: str, base_model: str, args: argparse.Namespace, logger: logging.Logger):
    """
    加载 LoRA 适配器：先加载基础模型，再挂载 PEFT 适配器；可选合并。
    """
    logger.info(f"检测到 LoRA 适配器目录，基础模型: {base_model}")
    base, _ = _FastModel.from_pretrained(
        model_name=base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        full_finetuning=False,
        token=None,
    )
    model = PeftModel.from_pretrained(base, ckpt_dir)
    if args.merge_lora:
        logger.info("合并 LoRA 到基础权重（merge_and_unload）…")
        try:
            model = model.merge_and_unload()
        except Exception as e:
            logger.warning(f"合并失败（将保持 PEFT 形式运行）：{e}")
    return model


def build_infer_model_and_tokenizer(args: argparse.Namespace, logger: logging.Logger):
    """
    统一入口：根据目录类型加载 tokenizer + model
    """
    # 若传入的是训练输出根目录，优先下钻最新 checkpoint
    chosen_dir = args.ckpt_dir
    if _detect_ckpt_type(chosen_dir) == "unknown":
        latest = _find_latest_checkpoint(chosen_dir)
        if latest:
            logger.info(f"未识别到直接权重，已自动切换到最新 checkpoint: {latest}")
            chosen_dir = latest

    ctype = _detect_ckpt_type(chosen_dir)
    if ctype == "unknown":
        raise ValueError(f"无法在目录中识别有效的模型/适配器: {chosen_dir}")

    tokenizer = _load_tokenizer_from_dir_or_base(
        ckpt_dir=chosen_dir,
        base_model=args.base_model,
        chat_template=args.chat_template,
        logger=logger,
    )

    if ctype == "full":
        model = _load_full_model(chosen_dir, args, logger)
    else:
        model = _load_adapter_model(chosen_dir, args.base_model, args, logger)

    # 统一到 eval 模式
    model.eval()
    # 大多数情况下需要将推理张量送到 CUDA（Unsloth 已经处理）；保险起见：
    if torch.cuda.is_available():
        try:
            model.to("cuda")
        except Exception:
            pass

    return model, tokenizer


# ==============================
# 思维段落处理
# ==============================

_THINK_PATTERNS = [
    # 常见 <think>…</think>
    (re.compile(r"<\s*think\s*>[\s\S]*?<\s*/\s*think\s*>", re.IGNORECASE), ""),
    # Qwen3 Thinking 类模板：<|assistant_thought|> ... （直到下一个 <|im_start|>assistant 或结尾）
    (re.compile(r"<\|assistant_thought\|>[\s\S]*?(?=(<\|im_start\|>assistant|$))", re.IGNORECASE), ""),
    # 其它可能的思维标记
    (re.compile(r"<\s*reasoning\s*>[\s\S]*?<\s*/\s*reasoning\s*>", re.IGNORECASE), ""),
    (re.compile(r"【\s*思考过程\s*】[\s\S]*?【\s*/?思考过程\s*】", re.IGNORECASE), ""),
]

_SPECIAL_TOKS = [
    re.compile(r"<\|im_start\|>assistant\s*", re.IGNORECASE),
    re.compile(r"<\|im_end\|>\s*", re.IGNORECASE),
]


def strip_thinking_blocks(text: str) -> str:
    """剥离常见思维链标记块，保留最终答案文本。"""
    cleaned = text
    for pat, repl in _THINK_PATTERNS:
        cleaned = pat.sub(repl, cleaned)
    # 清理模板特殊 token（可选）
    for pat in _SPECIAL_TOKS:
        cleaned = pat.sub("", cleaned)
    # 常见残留分隔
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _extract_generated_segment(full_decoded: str, prompt_text: str) -> str:
    """从完整解码文本中抽取新生成的片段。与 inference_sft.py 行为一致。"""
    return full_decoded[len(prompt_text):]


# ==============================
# 推理执行
# ==============================

def run_inference(model, tokenizer, messages: List[dict], args: argparse.Namespace):
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # 生成必须加
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        # 可按需启用：避免与部分静态 KV-cache 不兼容
        # cache_implementation="dynamic",
    )

    streamer = None if args.no_stream else TextStreamer(tokenizer, skip_prompt=True)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            streamer=streamer,
            **gen_kwargs,
        )

    # 非流式情况下打印一次性输出
    if args.no_stream:
        decoded = tokenizer.decode(out[0], skip_special_tokens=False)
        gen_text = _extract_generated_segment(decoded, prompt_text)
        if args.strip_thought:
            gen_text = strip_thinking_blocks(gen_text)
        print(gen_text)


# ==============================
# 入口
# ==============================

def main():
    cfg_defaults = TrainConfig()  # 直接复用训练端默认配置，避免重复定义
    args = parse_args(cfg_defaults)

    logger = setup_logging(output_dir=os.path.join(args.ckpt_dir, "logs_infer_thinking"))
    set_seed(args.seed, logger)
    log_env_info(logger)

    model, tokenizer = build_infer_model_and_tokenizer(args, logger)
    # 如遇 transformers 静态 KV 与 PEFT/4bit 兼容问题，可启用：
    # model.generation_config.cache_implementation = "dynamic"

    messages = _load_messages(args)
    logger.info(f"推理消息: {messages}")

    run_inference(model, tokenizer, messages, args)
    logger.info("✅ 推理完成。")


if __name__ == "__main__":
    main()
