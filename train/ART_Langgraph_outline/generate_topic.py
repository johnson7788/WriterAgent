#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/12 15:40
# @File  : generate_topic.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :
"""
Med Topic Agent — generate one‑sentence medical research topics and save to JSON.

Quickstart
----------
1) Install
   pip install openai-agents pydantic

2) .env文件，设置OpenAI的key

3) Run
   python med_topic_agent.py -n 20 -o topics.json --lang zh
"""
from __future__ import annotations

import argparse
import json
import sys
import dotenv
from pathlib import Path
from typing import List, Set
from pydantic import BaseModel, Field
from agents import Agent, Runner
dotenv.load_dotenv()

# ---------- Structured output schema ----------
class TopicsOutput(BaseModel):
    """Final structured output for the agent.

    Fields
    ------
    topics: A list of unique, one‑sentence medical research topics in the requested language.
    """

    topics: List[str] = Field(
        description=(
            "A list of unique, one‑sentence medical research topics. "
            "Each item must be one sentence, specific (not vague), and suitable as an academic research theme."
        )
    )


# ---------- Agent factory ----------
def build_agent(language: str) -> Agent:
    lang_prompt = {
        "zh": "中文",
        "en": "English",
    }.get(language, language)

    instructions = f"""
你是“医学研究选题助手”。
目标：根据用户给定的数量 N，生成 N 条“不同的、学术化的、一句话”医学研究主题，使用 {lang_prompt} 输出。
严格要求：
- 每条必须是一句话（不分点、不编号、不加前缀数字/符号）。
- 具体清晰，尽量包含：目标人群/疾病/场景/方法/指标/数据来源中的 ≥2 个元素。
- 主题之间必须互不重复、语义不近似。
- 直接遵循输出模式（structured outputs），不要解释或添加额外文本。
"""

    return Agent(
        name="MedTopicAgent",
        instructions=instructions.strip(),
        # Use structured outputs for reliability
        output_type=TopicsOutput,
        # You may set `model` or `model_settings` here if you want
        model="gpt-4.1",  # example: faster model if available
    )


# ---------- Generation helpers ----------

def _normalize(s: str) -> str:
    # Minimal normalization for de-duplication
    return "".join(s.strip().split()).lower()


def generate_unique_topics(agent: Agent, n: int, language: str, max_rounds: int = 3) -> List[str]:
    """Ask the agent to produce `n` unique topics; if duplicates slip through,
    re-ask for the remaining count up to `max_rounds` times.
    """
    unique: List[str] = []
    seen: Set[str] = set()
    remaining = n

    for round_idx in range(max_rounds):
        if remaining <= 0:
            break

        # Provide an explicit instruction with remaining count and an avoid list
        avoid_note = (
            "\n请避免与以下主题重复或过于相似：\n- " + "\n- ".join(unique)
            if unique
            else ""
        )

        prompt = (
            f"请严格生成 {remaining} 条不同的一句话医学研究主题，使用 {('中文' if language=='zh' else 'English')}。"
            f"{avoid_note}\n只需返回 topics 列表（structured output）。"
        )

        result = Runner.run_sync(agent, prompt)
        topics_batch = result.final_output.topics  # type: ignore[attr-defined]

        # Deduplicate and keep insertion order
        for t in topics_batch:
            key = _normalize(t)
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(t.strip())
            if len(unique) >= n:
                break

        remaining = n - len(unique)

    if len(unique) < n:
        raise RuntimeError(
            f"Only generated {len(unique)} unique topics after {max_rounds} rounds; try increasing -n fewer,"
            " or raise max_rounds."
        )

    return unique


# ---------- Main CLI ----------

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate medical research topics to JSON via OpenAI Agents SDK.")
    parser.add_argument("-n", "--num", type=int, default=50, help="Number of topics to generate (N ≥ 1)")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("topic.json"),
        help="Output JSON file path (default: ./topic.json)",
    )
    parser.add_argument(
        "--lang",
        choices=["zh", "en"],
        default="zh",
        help="Language of topics: zh (Chinese) or en (English). Default: zh",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Max extra rounds to top up uniques if duplicates slip through (default: 3)",
    )

    args = parser.parse_args(argv)
    if args.num < 1:
        parser.error("--num must be ≥ 1")

    agent = build_agent(args.lang)
    topics = generate_unique_topics(agent, args.num, language=args.lang, max_rounds=args.max_rounds)

    payload = {"language": args.lang, "count": len(topics), "topics": topics}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Wrote {len(topics)} topics to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
