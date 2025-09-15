#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/7 21:36
# @File  : inference_GRPO.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/09/07
# @File  : inference_GRPO.py
# @Author: johnson
# @Desc  : 推理脚本（适配 train_GRPO.py 的 GRPO 训练产物）
"""
用法示例（与 train_grpo.py 默认保存目录一致）：

1) LoRA 目录直接推理（训练脚本保存到 ./outputs/grpo_saved_lora）
---------------------------------------------------------------
python inference_grpo.py \
  --ckpt_dir ./outputs/grpo_saved_lora \
  --base_model unsloth/Qwen3-4B-Base \
  --prompt "一道经典鸡兔同笼题：鸡和兔共有 20 个头，50 条腿，各有多少只？" \
  --max_new_tokens 256

2) 自动定位输出根目录下“最新 checkpoint-*”
---------------------------------------------------------------
python inference_grpo.py \
  --ckpt_dir ./outputs \
  --prompt "Solve: (x-3)^2 = 25" \
  --no_stream

3) 合并 LoRA（更快推理）
---------------------------------------------------------------
python inference_grpo.py \
  --ckpt_dir ./outputs/grpo_saved_lora \
  --base_model unsloth/Qwen3-4B-Base \
  --merge_lora \
  --prompt "Explain breadth-first search."

4) 仅输出 <SOLUTION> ... </SOLUTION> 中的最终答案（非流式）
---------------------------------------------------------------
python inference_GRPO.py \
  --ckpt_dir ./outputs/grpo_saved_lora \
  --prompt "2025 年 3 月共有多少个星期日？" \
  --solution_only --no_stream

5) 剥离思维过程（<start_working_out> ... <end_working_out>），保留其余文本（非流式）
---------------------------------------------------------------
python inference_GRPO.py \
  --ckpt_dir ./outputs/grpo_saved_lora \
  --prompt "请给一道中考几何压轴题并详细解答。" \
  --strip_reasoning --no_stream
"""

from __future__ import annotations
import os
import re
import json
import argparse
import logging
from typing import List, Optional

import torch
from transformers import AutoTokenizer, TextStreamer

# 复用你已有的训练公共模块（与 inference_sft.py / inference_thinking.py 一致）
from unsloth_core import (
    TrainConfig,
    setup_logging,
    set_seed,
    log_env_info,
)

# Unsloth 加速加载
from unsloth import FastModel as _FastModel
from peft import PeftModel


# ==============================
# 与 train_GRPO.py 一致的标签 & 系统提示
# ==============================
DEFAULT_REASONING_START = "<start_working_out>"
DEFAULT_REASONING_END   = "<end_working_out>"
DEFAULT_SOLUTION_START  = "<SOLUTION>"
DEFAULT_SOLUTION_END    = "</SOLUTION>"

DEFAULT_SYSTEM_PROMPT = (
    "You are given a problem.\n"
    "Think about the problem and provide your working out.\n"
    f"Place it between {DEFAULT_REASONING_START} and {DEFAULT_REASONING_END}.\n"
    f"Then, provide your solution between {DEFAULT_SOLUTION_START}{DEFAULT_SOLUTION_END}"
)

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
        for fname in [
            "model.safetensors",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
            "model.safetensors.index.json",
        ]:
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
# CLI 参数
# ==============================
def parse_args(cfg_defaults: TrainConfig) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO 推理脚本（适配 train_GRPO.py）")

    # 权重与设备
    p.add_argument("--ckpt_dir", type=str, default=cfg_defaults.output_dir,
                   help="训练产物目录（可为最终输出目录、checkpoint-* 子目录或 LoRA 适配器目录）")
    p.add_argument("--base_model", type=str, default="unsloth/Qwen3-4B-Base",
                   help="当 --ckpt_dir 为 LoRA 适配器目录时用于加载的基座")
    p.add_argument("--max_seq_length", type=int, default=cfg_defaults.max_seq_length)
    p.add_argument("--load_in_4bit", action="store_true", default=cfg_defaults.load_in_4bit)
    p.add_argument("--no_load_in_4bit", action="store_true")
    p.add_argument("--load_in_8bit", action="store_true", default=cfg_defaults.load_in_8bit)
    p.add_argument("--no_load_in_8bit", action="store_true")
    p.add_argument("--merge_lora", action="store_true", help="合并 LoRA 并卸载适配器（推理更快）")

    # 模板/随机种子
    p.add_argument("--force_grpo_template", action="store_true",
                   help="强制覆盖当前 tokenizer.chat_template 为 GRPO 模板（当 checkpoint 未保存模板时很有用）")
    p.add_argument("--seed", type=int, default=cfg_defaults.seed)

    # 生成参数
    p.add_argument("--prompt", type=str, default=None,
                   help="单条 user 提示词。若提供则覆盖默认示例")
    p.add_argument("--messages_json", type=str, default=None,
                   help="包含 messages(list[{'role','content'}]) 的 JSON 文件路径")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--no_stream", action="store_true", help="关闭流式输出（开启后才会进行后处理）")

    # GRPO 标签参数（如需自定义）
    p.add_argument("--reasoning_start", type=str, default=DEFAULT_REASONING_START)
    p.add_argument("--reasoning_end",   type=str, default=DEFAULT_REASONING_END)
    p.add_argument("--solution_start",  type=str, default=DEFAULT_SOLUTION_START)
    p.add_argument("--solution_end",    type=str, default=DEFAULT_SOLUTION_END)
    p.add_argument("--system_prompt",   type=str, default=DEFAULT_SYSTEM_PROMPT)

    # 输出控制
    p.add_argument("--strip_reasoning", action="store_true",
                   help="剥离 <start_working_out>…<end_working_out> 段，仅保留其余文本（需 --no_stream）")
    p.add_argument("--solution_only", action="store_true",
                   help="仅打印 <SOLUTION>…</SOLUTION> 的最终答案（需 --no_stream）")

    args = p.parse_args()

    # 布尔修正
    if args.no_load_in_4bit:
        args.load_in_4bit = False
    if args.no_load_in_8bit:
        args.load_in_8bit = False
    if args.load_in_4bit and args.load_in_8bit:
        args.load_in_8bit = False  # 4bit 优先

    # 输出模式互斥提示
    if args.solution_only and args.strip_reasoning:
        print("[警告] --solution_only 与 --strip_reasoning 同时指定时，仅生效 --solution_only。")

    # solution_only / strip_reasoning 需要一次性输出文本
    if (args.solution_only or args.strip_reasoning) and not args.no_stream:
        args.no_stream = True

    return args


# ==============================
# 构建与训练一致的 Chat Template
# ==============================
def build_grpo_chat_template(tokenizer, system_prompt: str, reasoning_start: str):
    """
    与 train_GRPO.py 的 build_chat_template 等价：
    - 若首条不是 system，则自动注入 system_prompt + eos_token
    - add_generation_prompt=True 时，在末尾插入 <start_working_out>
    """
    chat_template = (
        "{% if messages[0]['role'] == 'system' %}"
            "{{ messages[0]['content'] + eos_token }}"
            "{% set loop_messages = messages[1:] %}"
        "{% else %}"
            "{{ '" + system_prompt.replace("'", "\\'") + "' + eos_token }}"
            "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
            "{% if message['role'] == 'user' %}"
                "{{ message['content'] }}"
            "{% elif message['role'] == 'assistant' %}"
                "{{ message['content'] + eos_token }}"
            "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '" + reasoning_start.replace("'", "\\'") + "' }}{% endif %}"
    )
    tokenizer.chat_template = chat_template
    return tokenizer


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
        return [{"role": "user", "content": args.prompt.strip()}]
    # 默认示例（鼓励推理 + 最终答案）
    return [
        {"role": "user", "content": "解方程：2x + 3 = 15。请给出推理过程与最终答案。"}
    ]


# ==============================
# 加载 tokenizer / model
# ==============================
def _load_tokenizer(ckpt_dir_or_base: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(ckpt_dir_or_base, use_fast=True)


def _load_full_model(ckpt_dir: str, args: argparse.Namespace, logger: logging.Logger):
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
    - 优先使用 ckpt_dir 下保存的 tokenizer（train_GRPO.py 在保存 LoRA 时也保存了 tokenizer）
    - 若未保存 tokenizer，则回退到 base_model
    - 若 ckpt_dir 是输出根目录，自动下钻最新 checkpoint-*
    """
    chosen_dir = args.ckpt_dir
    if _detect_ckpt_type(chosen_dir) == "unknown":
        latest = _find_latest_checkpoint(chosen_dir)
        if latest:
            logger.info(f"未识别到直接权重，已自动切换到最新 checkpoint: {latest}")
            chosen_dir = latest

    ctype = _detect_ckpt_type(chosen_dir)
    if ctype == "unknown":
        raise ValueError(f"无法在目录中识别有效的模型/适配器: {chosen_dir}")

    # 优先从 ckpt_dir 读 tokenizer，否则退回 base_model
    tok_src = chosen_dir if (
        _file_exists(chosen_dir, "tokenizer.json") or _file_exists(chosen_dir, "tokenizer_config.json")
    ) else args.base_model

    logger.info(f"加载 Tokenizer 自: {tok_src}")
    tokenizer = _load_tokenizer(tok_src)

    # 如未保存模板或显式要求，覆盖为 GRPO 模板（确保与训练一致）
    need_force = args.force_grpo_template
    raw_tpl = getattr(tokenizer, "chat_template", None)
    if (raw_tpl is None) or (DEFAULT_REASONING_START not in raw_tpl) or need_force:
        logger.info("应用 GRPO Chat Template（复现训练时模板）")
        tokenizer = build_grpo_chat_template(
            tokenizer,
            system_prompt=args.system_prompt,
            reasoning_start=args.reasoning_start,
        )
    else:
        logger.info("使用 checkpoint 中已有的 chat_template。")

    # 加载模型
    if ctype == "full":
        model = _load_full_model(chosen_dir, args, logger)
    else:
        model = _load_adapter_model(chosen_dir, args.base_model, args, logger)

    model.eval()
    if torch.cuda.is_available():
        try:
            model.to("cuda")
        except Exception:
            pass

    return model, tokenizer


# ==============================
# 文本后处理
# ==============================
def _extract_generated_segment(full_decoded: str, prompt_text: str) -> str:
    return full_decoded[len(prompt_text):]


def strip_reasoning_blocks(text: str, start_tok: str, end_tok: str) -> str:
    """
    剥离 <start_working_out> ... <end_working_out> 段。
    """
    try:
        pat = re.compile(
            re.escape(start_tok) + r"[\s\S]*?" + re.escape(end_tok),
            flags=re.MULTILINE,
        )
        cleaned = pat.sub("", text)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned
    except Exception:
        return text


def extract_solution_only(text: str, sol_start: str, sol_end: str) -> str:
    """
    仅提取 <SOLUTION> ... </SOLUTION> 内容；若未匹配到，则回退原文。
    """
    m = re.search(
        re.escape(sol_start) + r"([\s\S]+?)" + re.escape(sol_end),
        text,
        flags=re.MULTILINE,
    )
    if m:
        return m.group(1).strip()
    return text.strip()


# ==============================
# 推理执行
# ==============================
def run_inference(model, tokenizer, messages: List[dict], args: argparse.Namespace):
    # 注意：模板会在首条非 system 的情况下自动注入 system_prompt
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # 必须：在末尾注入 <start_working_out>
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        # 如遇静态 KV 与 4bit/PEFT 兼容问题可启用：
        # cache_implementation="dynamic",
    )

    streamer = None if args.no_stream else TextStreamer(tokenizer, skip_prompt=True)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            streamer=streamer,
            **gen_kwargs,
        )

    # 非流式：统一做后处理并打印
    if args.no_stream:
        decoded = tokenizer.decode(out[0], skip_special_tokens=False)
        gen_text = _extract_generated_segment(decoded, prompt_text)

        if args.solution_only:
            gen_text = extract_solution_only(gen_text, args.solution_start, args.solution_end)
        elif args.strip_reasoning:
            gen_text = strip_reasoning_blocks(gen_text, args.reasoning_start, args.reasoning_end)

        print(gen_text)


# ==============================
# 入口
# ==============================
def main():
    cfg_defaults = TrainConfig()  # 复用训练侧默认（max_seq_length/load_in_4bit 等）
    args = parse_args(cfg_defaults)

    logger = setup_logging(output_dir=os.path.join(args.ckpt_dir, "logs_infer_grpo"))
    set_seed(args.seed, logger)
    log_env_info(logger)

    model, tokenizer = build_infer_model_and_tokenizer(args, logger)

    messages = _load_messages(args)
    logger.info(f"推理消息: {messages}")

    run_inference(model, tokenizer, messages, args)
    logger.info("✅ GRPO 推理完成。")


if __name__ == "__main__":
    main()
