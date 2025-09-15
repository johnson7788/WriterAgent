#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/7 11:02
# @File  : GRPO_test.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 测试GPRO的环境是否正常

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train")

def reward_num_unique_letters(completions, **kwargs):
    texts = [c[0]["content"] for c in completions]
    return [float(len(set(t))) for t in texts]

args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO")  # 只做连通性验证
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_num_unique_letters,
    args=args,
    train_dataset=dataset,
)
print("GRPO ready.")
