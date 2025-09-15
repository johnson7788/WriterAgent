#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用 Unsloth + TRL 的 GRPO 在 open-r1/DAPO-Math-17k-Processed 数据集上训练推理格式与答案。
本脚本在原版基础上 **集成 Weights & Biases（wandb）**，将训练过程的关键信息、指标与样例输出记录到 wandb，并在训练结束后可选上传 LoRA 适配器为 W&B Artifact。

## 显卡1上启动vllm,方便进行使用
CUDA_VISIBLE_DEVICES=1 \
trl vllm-serve --model unsloth/Qwen3-4B-Base \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --gpu-memory-utilization 0.6 \
  --max-model-len 2048 \
  --host 127.0.0.1 --port 8000

环境依赖（示例）：
pip install -U "unsloth" "trl" "vllm" "datasets" "transformers==4.55.4" "bitsandbytes" "xformers" "torchvision" "pandas" wandb
"""

import os
import re
import gc
import math
import time
import json
import random
import logging
import argparse
import dotenv
from dataclasses import dataclass

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset

from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
dotenv.load_dotenv()
# ==== NEW: wandb 集成 ====
try:
    import wandb
    _WANDB_AVAILABLE = True
    print(f"WANDB_AVAILABLE是可用的，已经安装了wandb")
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False
    print(f"WANDB_AVAILABLE不可用，没有安装wandb")

# =========================
# 日志设置
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
)
logger = logging.getLogger("GRPO-DAPO-Math")

# =========================
# 工具函数
# =========================
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =========================
# 思维/答案标签与系统提示
# =========================
reasoning_start = "<start_working_out>"  # 类似 <think>
reasoning_end   = "<end_working_out>"    # 类似 </think>
solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"

system_prompt = (
    f"You are given a problem.\n"
    f"Think about the problem and provide your working out.\n"
    f"Place it between {reasoning_start} and {reasoning_end}.\n"
    f"Then, provide your solution between {solution_start}{solution_end}"
)
# =========================
# 构造 Chat Template（重要）
# =========================
def build_chat_template(tokenizer):
    chat_template = (
        "{% if messages[0]['role'] == 'system' %}"
            "{{ messages[0]['content'] + eos_token }}"
            "{% set loop_messages = messages[1:] %}"
        "{% else %}"
            "{{ '" + system_prompt + "' + eos_token }}"
            "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
            "{% if message['role'] == 'user' %}"
                "{{ message['content'] }}"
            "{% elif message['role'] == 'assistant' %}"
                "{{ message['content'] + eos_token }}"
            "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '" + reasoning_start + "' }}{% endif %}"
    )
    tokenizer.chat_template = chat_template
    return tokenizer

# =========================
# （可选）极小 SFT 预对齐
# =========================
def maybe_pre_sft(model, tokenizer, enable_pre_sft: bool, seed: int, wandb_args=None):
    if not enable_pre_sft:
        logger.info("跳过预SFT对齐（可通过 --enable-pre-sft true 开启）")
        return
    logger.info("开始进行一个小规模的SFT以对齐输出格式（可选步骤）...")
    from trl import SFTTrainer, SFTConfig

    ds = load_dataset("unsloth/OpenMathReasoning-mini", split="cot").to_pandas()
    ds = ds[["expected_answer", "problem", "generated_solution"]]

    # 尝试把答案转为数字，过滤失效项（与原示例一致）
    is_number = np.array(pd.to_numeric(ds["expected_answer"], errors="coerce").notnull())
    ds = ds.iloc[np.where(is_number)[0]]

    def format_dataset(x):
        # 去除原有 <think> 标签，贴合我们自定义的标签
        thoughts = x["generated_solution"].replace("<think>", "").replace("</think>", "").strip()
        final_prompt = reasoning_start + thoughts + reasoning_end + \
                       solution_start + x["expected_answer"] + solution_end
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x["problem"]},
            {"role": "assistant", "content": final_prompt},
        ]

    ds["Messages"] = ds.apply(format_dataset, axis=1)
    ds["text"] = tokenizer.apply_chat_template(ds["Messages"].values.tolist(), tokenize=False)
    from datasets import Dataset as HFDataset
    ds_hf = HFDataset.from_pandas(ds)

    report_to = "wandb" if wandb_args and wandb_args.get("enabled") else "none"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_hf,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=5,
            num_train_epochs=2,
            learning_rate=2e-4,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=seed,
            report_to=report_to,
        ),
    )
    trainer.train()
    del ds_hf; gc.collect()
    torch.cuda.empty_cache()
    logger.info("预SFT对齐完成。")

# =========================
# 奖励函数
# =========================
def build_reward_funcs(tokenizer, wandb_enabled=False, wandb_log_samples=False):
    # 允许 </SOLUTION> 后有可选的 EOS
    solution_end_regex = r"</SOLUTION>[\s]{0,}(?:" + re.escape(tokenizer.eos_token) + ")?"

    # 匹配：... <end_working_out> <SOLUTION>答案</SOLUTION> [EOS]
    match_format = re.compile(
        rf"{re.escape(reasoning_end)}.*?"
        rf"{re.escape(solution_start)}(.+?){solution_end_regex}"
        rf"[\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL,
    )

    # 1) 完全匹配格式奖励
    def match_format_exactly(completions, **kwargs):
        scores = []
        for completion in completions:
            resp = completion[0]["content"]
            score = 3.0 if match_format.search(resp) is not None else 0.0
            scores.append(score)
        return scores

    # 2) 近似匹配格式奖励
    def match_format_approximately(completions, **kwargs):
        scores = []
        for completion in completions:
            resp = completion[0]["content"]
            score = 0.0
            score += 0.5 if resp.count(reasoning_end)   == 1 else -1.0
            score += 0.5 if resp.count(solution_start)  == 1 else -1.0
            score += 0.5 if resp.count(solution_end)    == 1 else -1.0
            scores.append(score)
        return scores

    # 3) 基于严格提取答案的奖励
    def check_answer(prompts, completions, answer, **kwargs):
        responses = [c[0]["content"] for c in completions]
        extracted = [
            m.group(1) if (m := match_format.search(r)) is not None else None
            for r in responses
        ]
        scores = []
        for guess, true_answer in zip(extracted, answer):
            score = 0.0
            if guess is None:
                scores.append(-2.0)
                continue
            if guess == true_answer:
                score += 5.0
            elif guess.strip() == true_answer.strip():
                score += 3.5
            else:
                try:
                    ratio = float(guess) / float(true_answer)
                    if 0.9 <= ratio <= 1.1:
                        score += 2.0
                    elif 0.8 <= ratio <= 1.2:
                        score += 1.5
                    else:
                        score -= 2.5
                except:
                    score -= 4.5
            scores.append(score)
        return scores

    # 4) 数字提取版（容忍“答案里带文字”，只抽取第一个数字进行对比）
    match_numbers = re.compile(
        re.escape(solution_start) + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
        flags=re.MULTILINE | re.DOTALL,
    )

    # 打印频率（便于观察模型输出）
    PRINT_EVERY_STEPS = 5
    printed_times = {"n": 0}

    def check_numbers(prompts, completions, answer, **kwargs):
        # 仅用第一个样本进行可读性日志打印
        question = prompts[0][-1]["content"]
        responses = [c[0]["content"] for c in completions]
        extracted = [
            m.group(1) if (m := match_numbers.search(r)) is not None else None
            for r in responses
        ]

        if printed_times["n"] % PRINT_EVERY_STEPS == 0:
            msg = ("\n" + "*"*20 +
                   f"\n【题目】\n{question}\n【标准答案】\n{answer[0]}" \
                   f"\n【模型输出】\n{responses[0]}" \
                   f"\n【提取数字】\n{extracted[0]}\n" + "*"*20)
            logger.info(msg)
            # === NEW: 同步到 wandb（可选） ===
            if wandb_enabled and wandb_log_samples and _WANDB_AVAILABLE and wandb.run is not None:
                # 为了简单与稳定，使用文本字段进行记录
                wandb.log({
                    "sample/question": question,
                    "sample/answer_true": str(answer[0]),
                    "sample/response": responses[0],
                    "sample/extracted_number": str(extracted[0]),
                })
        printed_times["n"] += 1

        scores = []
        for guess, true_answer in zip(extracted, answer):
            if guess is None:
                scores.append(-2.5)
                continue
            try:
                ta = float(true_answer.strip())
                ga = float(guess.strip().replace(",", ""))
                scores.append(3.5 if ga == ta else -1.5)
            except:
                scores.append(0.0)
        return scores

    return match_format_exactly, match_format_approximately, check_answer, check_numbers

# =========================
# 主流程
# =========================
def main(args):
    t0 = time.time()

    # 目录与种子
    os.makedirs(args.output_dir, exist_ok=True)
    lora_save_dir = args.lora_save_dir or os.path.join(args.output_dir, "grpo_saved_lora")
    logger.info(f"保存lora模型路径为: {lora_save_dir}")
    set_seed(args.seed)

    # ==== NEW: 初始化 W&B ====
    wandb_run = None
    wandb_enabled = bool(args.wandb and _WANDB_AVAILABLE and args.wandb_mode != "disabled")
    if args.wandb and not _WANDB_AVAILABLE:
        logger.warning("检测到 --wandb=true 但未安装 wandb；请先 `pip install wandb`。")
    if wandb_enabled:
        tags = [t.strip() for t in (args.wandb_tags or "").split(",") if t.strip()]
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            tags=tags,
            config={**{k:v for k,v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))}},
        )
        # 记录环境与硬件
        sys_info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "num_gpus": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        }
        wandb.config.update(sys_info, allow_val_change=True)

    # 1) 加载模型与Tokenizer（LoRA可训练）
    logger.info("加载基座模型与Tokenizer：%s", args.model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_len,
        load_in_4bit=args.load_in_4bit,
        fast_inference=args.fast_inference,
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=args.gpu_memory_util,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # 可选：监控梯度/权重（谨慎使用，可能较慢）
    if wandb_enabled and args.wandb_watch != "none":
        try:
            wandb.watch(model, log=args.wandb_watch, log_freq=max(50, args.logging_steps))
        except Exception as e:
            logger.warning(f"wandb.watch 失败：{e}")

    # 2) 设置 Chat Template
    tokenizer = build_chat_template(tokenizer)
    logger.info("Chat template 已设定完成。")

    # （可选）SFT 预对齐
    maybe_pre_sft(model, tokenizer, args.enable_pre_sft, args.seed,
                  wandb_args={"enabled": wandb_enabled})

    # 3) 加载数据集
    logger.info("加载数据集：%s (%s / %s)", args.hf_dataset, args.hf_config, args.hf_split)
    hf_name = None if (args.hf_config in [None, "", "none", "None", "-"]) else args.hf_config
    load_kwargs = {"split": args.hf_split}
    if args.data_files:
        data_files = args.data_files.split(",") if "," in args.data_files else args.data_files
        load_kwargs["data_files"] = data_files
        logger.info("使用 data_files: %s", data_files)
    ds = load_dataset(args.hf_dataset, hf_name, **load_kwargs)
    logger.info("数据集大小：%d", len(ds))
    logger.info("样例 prompt：%s", ds[0]["prompt"][:200].replace("\n", " "))
    logger.info("样例 solution：%s", ds[0]["solution"][:200].replace("\n", " "))

    # -------- 4) 数据字段映射到我们需要的格式 --------
    #    - prompt: 系统提示 + 用户题目
    #    - answer: 直接使用数据集的 "solution"（该数据集无需像 GSM8K 提取 #### 后的答案）
    def extract_hash_answer(text):
        # Open R1 这个处理版数据无需截 ####，保留原答案
        return text

    ds = ds.map(lambda x: {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": x["prompt"]},
        ],
        "answer": extract_hash_answer(x["solution"]),
    })
    logger.info("字段映射完成，示例：%s", json.dumps(ds[0]["prompt"][-1], ensure_ascii=False)[:300])

    # 5) 奖励函数（带 wandb 文本样例记录选项）
    reward_funcs = build_reward_funcs(tokenizer,
                                      wandb_enabled=wandb_enabled,
                                      wandb_log_samples=args.wandb_log_samples)
    logger.info("奖励函数已构建：%s", [f.__name__ for f in reward_funcs])

    # -------- 6) 统计长度并过滤超长样本（避免被截断影响训练） --------
    logger.info("开始统计 token 长度并过滤最长的 10%% 样本...")
    tokenized = ds.map(
        lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
        batched=True,
    )
    tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})
    Ls = np.array(tokenized["L"])
    max_prompt_len_q90 = int(np.quantile(Ls, 0.9))
    logger.info("90%% 分位的 prompt token 长度：%d", max_prompt_len_q90)
    kept_indices = np.where(Ls <= max_prompt_len_q90)[0]
    ds = ds.select(kept_indices.tolist())
    logger.info("过滤后数据集大小：%d（移除了 top 10%% 超长样本）", len(ds))

    # 7) 计算长度预算
    max_prompt_length = max_prompt_len_q90 + 1
    max_completion_length = args.max_seq_len - max_prompt_length
    logger.info("max_prompt_length=%d, max_completion_length=%d",
                max_prompt_length, max_completion_length)

    # === 将上述统计同步到 W&B ===
    if wandb_enabled:
        wandb.summary["dataset/size"] = len(ds)
        wandb.summary["prompt_len/q90"] = max_prompt_len_q90
        wandb.summary["budget/max_prompt_length"] = max_prompt_length
        wandb.summary["budget/max_completion_length"] = max_completion_length

    # 8) vLLM 采样参数
    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=args.seed,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )

    # 9) 训练参数（GRPO）
    report_to = "wandb" if wandb_enabled else "none"
    training_args = GRPOConfig(
        use_vllm = args.use_vllm,
        vllm_mode="server",
        vllm_server_base_url=args.vllm_server_base_url,
        vllm_sampling_params=vllm_sampling_params,
        temperature=1.0,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        optim=args.optim,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=args.max_steps,
        save_steps=args.save_steps if args.save_steps is not None else args.max_steps,
        report_to=report_to,
        output_dir=args.output_dir,
        # 评估相关参数可按需开启
        # fp16_full_eval=True,
        # per_device_eval_batch_size=4,
        # eval_accumulation_steps=1,
        # eval_strategy="steps",
        # eval_steps=50,
    )

    # 10) 启动 GRPO 训练
    logger.info("开始 GRPO 训练...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=list(reward_funcs),
        args=training_args,
        train_dataset=ds,
    )

    # 让 transformers 的 WandbCallback 也能介入（可选）
    if wandb_enabled:
        try:
            from transformers.integrations import WandbCallback
            trainer.add_callback(WandbCallback())
        except Exception as e:
            logger.warning(f"添加 WandbCallback 失败：{e}")

    trainer.train()
    logger.info("GRPO 训练完成。")

    # 11) 保存 LoRA 适配器
    logger.info("保存 LoRA 适配器到：%s", lora_save_dir)
    os.makedirs(lora_save_dir, exist_ok=True)
    trainer.save_model(lora_save_dir)
    tokenizer.save_pretrained(lora_save_dir)
    logger.info("模型已保存至: %s", lora_save_dir)

    # ===上传为 W&B Artifact（可选） ===
    if wandb_enabled and args.wandb_upload_artifact:
        try:
            safe_model_name = args.model_name.replace('/', '-').replace(' ', '_')
            artifact_name = f"{safe_model_name}-grpo-lora"
            art = wandb.Artifact(artifact_name, type="model",
                                 metadata={
                                     "model_name": args.model_name,
                                     "lora_rank": args.lora_rank,
                                     "max_steps": args.max_steps,
                                     "dataset": f"{args.hf_dataset}/{args.hf_config}:{args.hf_split}",
                                     "max_seq_len": args.max_seq_len,
                                 })
            art.add_dir(lora_save_dir)
            wandb.log_artifact(art)
            logger.info("LoRA 适配器已作为 Artifact 上传到 W&B：%s", artifact_name)
        except Exception as e:
            logger.warning(f"上传 Artifact 失败：{e}")

    elapsed_min = (time.time() - t0) / 60.0
    logger.info("全部完成，用时 %.1f 分钟。", elapsed_min)
    logger.info("提示：推理时可调用 model.load_lora('%s') 进行 LoRA 加载。", lora_save_dir)
    logger.info(f"保存lora模型路径为: {lora_save_dir}")

    if wandb_enabled:
        wandb.summary["elapsed/minutes"] = round(elapsed_min, 2)
        wandb.finish()

# =========================
# 参数解析
# =========================
def build_parser():
    p = argparse.ArgumentParser(description="GRPO 训练脚本（可通过命令行传参），已集成 W&B 记录")

    # 模型 & 训练基础
    p.add_argument("--model-name", type=str, default="unsloth/Qwen3-4B-Base")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--gpu-memory-util", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=3407)

    # 训练超参
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--optim", type=str, default="adamw_8bit", choices=["adamw_8bit", "adamw_torch", "adamw_hf"])
    p.add_argument("--save-steps", type=int, default=None, help="不指定则与 max_steps 相同")
    p.add_argument("--logging-steps", type=int, default=1, help="wandb/日志记录步频")

    # 数据集
    p.add_argument("--hf-dataset", type=str, default="open-r1/DAPO-Math-17k-Processed")
    p.add_argument("--hf-config",  type=str, default="en")
    p.add_argument("--hf-split",   type=str, default="train")
    # 自定义数据， 典型用法：--hf-dataset json --hf-config none --hf-split train --data_files /path/to/data.jsonl
    p.add_argument("--data-files", type=str, default=None,help = "自定义数据文件路径或通配（可逗号分隔）。与 --hf-dataset=json 等通用构建器配合使用。")

    # 预对齐
    p.add_argument("--enable-pre-sft", type=str2bool, default=False)

    # 目录
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--lora-save-dir", type=str, default=None, help="默认 outputs/grpo_saved_lora")

    # vLLM / 加速相关
    p.add_argument("--use-vllm", type=str2bool, default=True)
    p.add_argument("--vllm-server-base-url", type=str, default="http://127.0.0.1:8000")
    p.add_argument("--load-in-4bit", type=str2bool, default=False)
    p.add_argument("--fast-inference", type=str2bool, default=False)

    # ===== NEW: wandb 相关 =====
    p.add_argument("--wandb", type=str2bool, default=True, help="启用 Weights & Biases 记录")
    p.add_argument("--wandb-project", type=str, default="grpo-dapo-math", help="W&B 项目名")
    p.add_argument("--wandb-entity", type=str, default=None, help="W&B 实体（团队/用户名），可不填")
    p.add_argument("--wandb-run-name", type=str, default=None, help="W&B 运行名，不填则自动生成")
    p.add_argument("--wandb-tags", type=str, default="dapo,grpo,unsloth", help="逗号分隔的标签")
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"],
                   help="online=联网记录，offline=离线，disabled=完全关闭")
    p.add_argument("--wandb-log-samples", type=str2bool, default=True,
                   help="在奖励函数中按频率记录样例问答文本，便于排查训练质量")
    p.add_argument("--wandb-upload-artifact", type=str2bool, default=True, help="训练结束上传 LoRA 适配器为 Artifact")
    p.add_argument("--wandb-watch", type=str, default="none", choices=["none", "gradients", "all"],
                   help="是否使用 wandb.watch 监控梯度/权重（可能较慢）")

    return p

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
