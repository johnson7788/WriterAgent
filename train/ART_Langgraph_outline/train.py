#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/27 11:30
# @File  : train.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 训练“按 topic 搜索并生成 Markdown 大纲”的 LangGraph ReAct Agent（ART GRPO）
import logging
logging.basicConfig(level=logging.DEBUG)
import os
import uuid
import time
import asyncio
from statistics import mean
from textwrap import dedent
from typing import List, Optional
import re
import json
import dotenv
import prompt
import art
from art.langgraph import init_chat_model, wrap_rollout
from art.utils import iterate_dataset
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt
from litellm import acompletion
from zai import ZhipuAiClient
from art.rewards import ruler_score_group
from transformers import AutoTokenizer

# ---------------- wandb ----------------
import wandb

dotenv.load_dotenv()

# ---------- 配置 ----------
NAME = os.getenv("ART_NAME", "web-search-outline")
MODEL_NAME = os.getenv("ART_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROJECT_NAME = os.getenv("ART_PROJECT", "web-search-outline-training")
USE_LOCAL_BACKEND = os.getenv("ART_BACKEND", "local").lower() == "local"
USE_RULER = os.getenv("USE_RULER", "true").lower() == "true"
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", 4096))

print(f"{NAME} - {MODEL_NAME} - {PROJECT_NAME} - {os.environ['WANDB_BASE_URL']} - 很关键的USE_RULER: {USE_RULER}")
print(f"训练时传入的最大序列长度: {MAX_SEQ_LEN}")

# RULER 评估模型（可选；需相应 API Key）
RULER_MODEL = os.getenv("RULER_MODEL", "openai/o4-mini")

# wandb
WANDB_PROJECT = os.getenv("WANDB_PROJECT", PROJECT_NAME)
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_RUN_NAME = os.getenv("WANDB_RUN_NAME", f"{NAME}-{time.strftime('%Y%m%d-%H%M%S')}")

WebSearchClient = ZhipuAiClient(api_key=os.environ["ZHIPU_API_KEY"])

tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# --- 用于裁剪Context，当长度比较长的时候
def _msg_text(m):
    """将各种消息对象（dict / LangChain Message / OpenAI Choice / 其它）统一成 'role: content' 文本。"""
    # 1) dict 消息：用 dict.get
    if isinstance(m, dict):
        role = m.get("role", "") or m.get("type", "")
        content = m.get("content", "") or ""
        return f"{role or 'msg'}: {content}"

    # 2) OpenAI ChatCompletion Choice（或类似对象）：有 message 且 message.content
    #    采用鸭子类型判断，避免显式依赖 openai 的类型
    if hasattr(m, "message") and hasattr(getattr(m, "message"), "content"):
        role = "assistant"
        content = getattr(getattr(m, "message"), "content", "") or ""
        return f"{role}: {content}"

    # 3) 其它（如 LangChain 的 HumanMessage/SystemMessage 等）
    role = getattr(m, "type", None) or getattr(m, "role", "") or ""
    content = getattr(m, "content", "") or ""
    return f"{role or 'msg'}: {content}"

def _tokens_len(text: str) -> int:
    return len(tok(text, add_special_tokens=False, return_attention_mask=False)["input_ids"])

def clip_traj_inplace(traj, max_tokens=MAX_SEQ_LEN):
    # 对rollout的轨迹进行裁剪，保留一定长度即可，裁剪单条轨迹
    if not getattr(traj, "messages_and_choices", None):
        return
    msgs = list(traj.messages_and_choices)
    print(f"裁剪前有信息：{len(msgs)} 条")
    # 永远保留第一个 system（如有）
    keep_head = []
    if msgs and ("system" in _msg_text(msgs[0]).lower()):
        keep_head.append(msgs.pop(0))

    # 从“最近”往回累加，超出则停止
    kept_tail = []
    for m in reversed(msgs):
        candidate = keep_head + list(reversed(kept_tail + [m]))
        text = "\n".join(_msg_text(x) for x in candidate)
        if _tokens_len(text) <= max_tokens:
            kept_tail.append(m)
        else:
            break

    traj.messages_and_choices = keep_head + list(reversed(kept_tail))


# 在 finished / judged 生成之后、train 之前
def clip_group(g, max_tokens=MAX_SEQ_LEN):
    return art.TrajectoryGroup(
        (clip_traj_inplace(t, max_tokens) or t) for t in list(g)
    )

# ---------- 数据结构 ----------
class WebSearchResult(BaseModel):
    url: str
    title: str
    snippet: str

class FinalOutline(BaseModel):
    outline: str
    source_urls: List[str]

class Scenario(BaseModel):
    id: str
    topic: str
    # 参考答案可留空；训练以相对评分/结构校验为主
    reference_outline: Optional[str] = None

class WebSearchScenario(BaseModel):
    step: int
    scenario: Scenario

class ProjectTrajectory(art.Trajectory):
    final_outline: Optional[FinalOutline] = None

# ---------- 搜索 ----------
async def search_web(keyword: str) -> List[WebSearchResult]:
    response = WebSearchClient.web_search.web_search(
        search_engine="search_std",
        search_query=keyword,
        count=4,
        search_recency_filter="noLimit",
        content_size="medium"
    )
    if not response.search_result:
        return []

    return [
        WebSearchResult(
            url=sr.link,
            title=sr.title,
            snippet=sr.content
        )
        for sr in response.search_result
    ]

# ---------- 简单结构打分（可选：用于日志） ----------
def _structure_score_cn(md: str) -> float:
    """
    纯规则校验，返回 [0,1] 分：
    - # 1个
    - ## 恰好5个
    - 每个 ## 下有 3-4 个 ###
    - 每个 ### 下有 3-5 个 “- ”要点行
    - 不含“参考”“结语”“目录”等额外段落
    """
    try:
        # 仅一行一级标题
        h1 = re.findall(r"(?m)^# [^\n]+$", md)
        if len(h1) != 1:
            return 0.0
        # 二级标题
        h2_positions = [(m.start(), m.group()) for m in re.finditer(r"(?m)^## [^\n]+$", md)]
        if len(h2_positions) != 5:
            return 0.2
        h2_positions.append((len(md), ""))  # 便于切片
        per_h2_pass = 0
        total_h3 = 0
        total_bullets = 0

        for i in range(5):
            start = h2_positions[i][0]
            end = h2_positions[i+1][0]
            block = md[start:end]

            h3s = list(re.finditer(r"(?m)^### [^\n]+$", block))
            if len(h3s) < 3 or len(h3s) > 4:
                continue
            # 统计每个 h3 下的要点数量
            ok_h3 = 0
            for j, h3 in enumerate(h3s):
                b_start = h3.end()
                b_end = h3s[j+1].start() if j+1 < len(h3s) else len(block)
                sub = block[b_start:b_end]
                bullets = re.findall(r"(?m)^- [^\n]+$", sub)
                # 约束：3-5条，短句，动词开头，不以句号结尾
                if 3 <= len(bullets) <= 5 and all(
                    len(b) <= len("- ") + 18 and not b.endswith(("。", ".", "．"))
                    for b in [x[2:] for x in bullets]  # 去掉 "- "
                ):
                    ok_h3 += 1
                    total_bullets += len(bullets)
            if ok_h3 == len(h3s):
                per_h2_pass += 1
            total_h3 += len(h3s)

        score = 0.2  # 通过基本检查
        score += 0.3 * (per_h2_pass / 5.0)
        score += 0.3 * min(1.0, total_h3 / 18.0)  # 期望 5*(3~4)=15~20
        score += 0.2 * min(1.0, total_bullets / 60.0)  # 粗略目标 20*3=60 起步
        # 负面关键词惩罚
        if re.search(r"(参考|结语|总结|目录|说明|免责声明)", md):
            score *= 0.7
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0

class CorrectnessJudgeResponse(BaseModel):
    reasoning: str = Field(description="why")
    accept: bool = Field(description="是否满足结构与中文规范")

@retry(stop=stop_after_attempt(3))
async def judge_correctness(s: Scenario, outline: str) -> CorrectnessJudgeResponse:
    """
    使用一个小模型进行结构性与格式性判断（可选，仅做日志展示）。
    """
    # system_prompt = prompt.JUDGE_SYSTEM_PROMPT
    # user = prompt.JUDGE_USER_PROMPT.format(
    #     topic=s.topic,
    #     outline=outline,
    # )
    # try:
    #     resp = await acompletion(
    #         model=os.environ["JUDGE_MODEL_NAME"],
    #         base_url=os.environ["JUDGE_MODEL_BASE_URL"],
    #         messages=[{"role": "system", "content": system_prompt},
    #                   {"role": "user", "content": user}],
    #         response_format=CorrectnessJudgeResponse,
    #     )
    #     return CorrectnessJudgeResponse.model_validate_json(
    #         resp.choices[0].message.content or "{}"
    #     )
    # except Exception:
    # 回退到规则打分
    score = _structure_score_cn(outline)
    return CorrectnessJudgeResponse(reasoning=f"rule_score={score:.2f}", accept=score >= 0.7)

# ---------- rollout：LangGraph + Tools ----------
async def rollout(model: art.Model, web_search_scenario: WebSearchScenario) -> ProjectTrajectory:
    scenario = web_search_scenario.scenario
    MAX_TURNS = 10

    traj = ProjectTrajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={"scenario_id": scenario.id, "step": web_search_scenario.step},
    )
    final_outline: Optional[FinalOutline] = None

    @tool
    async def web_search_tool(query: str) -> List[dict]:
        """进行网络搜索并返回结果列表。"""
        print(f"[tool:web_search] scenario_id={scenario.id} step={web_search_scenario.step} query={query}")
        results = await search_web(query)
        print(f"[tool:web_search] results={results}")
        return [r.model_dump() for r in results]

    @tool
    def return_final_outline_tool(outline: str, source_urls: List[str]) -> dict:
        """提交最终大纲与来源链接。"""
        nonlocal final_outline
        final_outline = FinalOutline(outline=outline, source_urls=source_urls)
        return final_outline.model_dump()

    tools = [search_web, return_final_outline_tool]

    # 用 ART 的 init_chat_model 注入可训练聊天模型
    chat_model = init_chat_model(MODEL_NAME, temperature=0.4)
    agent = create_react_agent(chat_model, tools)
    print(f"[rollout] START scenario_id={scenario.id} step={web_search_scenario.step} topic={scenario.topic}")

    await agent.ainvoke(
        {
            "messages": [
                SystemMessage(content=prompt.ROLLOUT_SYSTEM_PROMPT),
                HumanMessage(content=prompt.ROLLOUT_USER_PROMPT.format(topic=scenario.topic))
            ]
        },
        config={"configurable": {"thread_id": str(uuid.uuid4())},
                "recursion_limit": MAX_TURNS},
    )
    # USE_RULER=False 时，这个 traj.reward 会直接参与训练；
    # USE_RULER=True 时，这里设的 reward 会被后面 ruler_score_group(...) 生成的相对分所替换
    if final_outline:
        traj.final_outline = final_outline
        try:
            judge = await judge_correctness(scenario, final_outline.outline)
            traj.metrics["pass_structure"] = float(judge.accept)
            traj.metrics["rule_score"] = _structure_score_cn(final_outline.outline)
            traj.reward = traj.metrics["rule_score"]
        except Exception:
            # 兜底：至少用纯规则分作为奖励
            rs = _structure_score_cn(final_outline.outline)
            traj.metrics["rule_score"] = rs
            traj.reward = rs
    else:
        # 没有提交最终大纲的 rollout 给个低分，鼓励完成轨迹
        traj.reward = 0.0
    return traj

# ---------------- wandb: 日志封装 ----------------
def _log_batch_to_wandb(*, batch, finished_groups, use_ruler: bool):
    trajectories = []
    for g in finished_groups:
        if hasattr(g, "trajectories"):
            trajectories.extend(getattr(g, "trajectories"))
        else:
            try:
                trajectories.extend(list(g))
            except Exception:
                pass

    num_traj = len(trajectories)
    num_with_final = sum(1 for t in trajectories if getattr(t, "final_outline", None))
    pass_vals = []
    rule_scores = []
    for t in trajectories:
        m = getattr(t, "metrics", None)
        if isinstance(m, dict):
            if "pass_structure" in m:
                try:
                    pass_vals.append(float(m["pass_structure"]))
                except Exception:
                    pass
            if "rule_score" in m:
                try:
                    rule_scores.append(float(m["rule_score"]))
                except Exception:
                    pass

    pass_rate = mean(pass_vals) if pass_vals else 0.0
    avg_rule = mean(rule_scores) if rule_scores else 0.0
    coverage = (num_with_final / num_traj) if num_traj else 0.0

    try:
        table = wandb.Table(columns=["scenario_id", "topic", "outline_preview", "sources"])
        for t in trajectories[:40]:
            meta = getattr(t, "metadata", {}) or {}
            s_id = meta.get("scenario_id", "")
            topic = ""
            try:
                for s in batch.items:
                    if s.id == s_id:
                        topic = s.topic
                        break
            except Exception:
                pass
            fo = getattr(t, "final_outline", None)
            outline_preview = (getattr(fo, "outline", "") or "")[:500]
            srcs = ", ".join(getattr(fo, "source_urls", []) if fo else [])
            table.add_data(s_id, topic, outline_preview, srcs)
    except Exception:
        table = None

    log_dict = {
        "train/step": batch.step,
        "train/epoch": batch.epoch,
        "ruler/enabled": int(bool(use_ruler)),
        "data/num_trajectories": num_traj,
        "data/final_outline_coverage": coverage,
        "eval/pass_structure_rate": pass_rate,
        "eval/rule_score_avg": avg_rule,
    }
    if table is not None:
        log_dict["samples/rollouts"] = table

    wandb.log(log_dict, step=batch.step)

# ---------- 训练主程序 ----------
async def main():
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY if WANDB_ENTITY else None,
        name=WANDB_RUN_NAME,
        config={
            "art_project": PROJECT_NAME,
            "art_name": NAME,
            "base_model": MODEL_NAME,
            "backend": "local" if USE_LOCAL_BACKEND else "skypilot",
            "ruler_model": RULER_MODEL,
        },
        settings=wandb.Settings(start_method="thread"),
    )
    wandb.define_metric("*", step_metric="train/step")

    if USE_LOCAL_BACKEND:
        from art.local.backend import LocalBackend
        backend = LocalBackend()
    else:
        from art.skypilot.backend import SkyPilotBackend
        backend = await SkyPilotBackend.initialize_cluster(
            cluster_name=os.getenv("ART_SKYPILOT_CLUSTER", "art-cluster"),
            gpu=os.getenv("ART_GPU", "A100"),
        )

    model = art.TrainableModel(name=NAME, project=PROJECT_NAME, base_model=MODEL_NAME)
    await model.register(backend)

    # 训练集：从 topic.json 文件加载
    assert os.path.exists('topic.json'), "训练的主题数据不存在，请检查topic.json文件"
    with open('topic.json', 'r', encoding='utf-8') as f:
        topic_data = json.load(f)
    training_scenarios = [
        Scenario(id=str(i), topic=topic)
        for i, topic in enumerate(topic_data["topics"], 1)
    ]

    # 训练参数， max_steps和num_epochs的区别
    training_config = {
        "groups_per_step": 2,
        "num_epochs": 2,
        "rollouts_per_group": 4,
        "learning_rate": 1e-5,
        "max_steps": int(os.environ.get("MAX_STEPS", 10)),
    }

    # wandb 数据概览
    try:
        scen_table = wandb.Table(columns=["id", "topic"])
        for s in training_scenarios:
            scen_table.add_data(s.id, s.topic)
        wandb.log({"data/training_scenarios": scen_table}, step=0)
    except Exception:
        pass

    it = iterate_dataset(
        training_scenarios,
        groups_per_step=training_config["groups_per_step"],
        num_epochs=training_config["num_epochs"],
        initial_step=await model.get_step(),
    )

    # 是否使用 RULER（若不可用会自动回退到相对比较）

    for batch in it:
        print(f"[train] step={batch.step} epoch={batch.epoch}")

        groups = []
        for s in batch.items:
            groups.append(
                art.TrajectoryGroup(
                    wrap_rollout(model, rollout)(model, WebSearchScenario(step=batch.step, scenario=s))
                    for _ in range(training_config["rollouts_per_group"])
                )
            )

        finished = await art.gather_trajectory_groups(
            groups, pbar_desc="gather",
            max_exceptions=training_config["rollouts_per_group"] * len(batch.items),
        )

        _log_batch_to_wandb(batch=batch, finished_groups=finished, use_ruler=USE_RULER)

        if USE_RULER:
            extra_litellm_params = {"api_base": "http://localhost:6688", "api_key": os.environ["OPENAI_API_KEY"]}
            judged = []
            for g in finished:
                t_list = list(g)
                completed = [t for t in t_list if getattr(t, "final_outline", None)]
                try:
                    # 完成数如果太少，那么就使用原始的reward打分结果
                    if len(completed) >= 2:
                        jg = await ruler_score_group(
                            art.TrajectoryGroup(completed),
                            RULER_MODEL,
                            extra_litellm_params=extra_litellm_params,
                            debug=True
                        )
                        judged.append(jg)
                    else:
                        # 完成数太少：直接用原始（含你在 rollout 里设的 reward）
                        judged.append(art.TrajectoryGroup(t_list))
                except Exception:
                    # RULER 失效/异常时，退回无裁判训练
                    judged.append(art.TrajectoryGroup(t_list))
            judged = [clip_group(g, MAX_SEQ_LEN) for g in judged]
            await model.train(
                trajectory_groups=judged,
                config=art.TrainConfig(learning_rate=training_config["learning_rate"]),
                _config={"logprob_calculation_chunk_size": 8},
            )
            wandb.log({"train/used_judged_groups": 1}, step=batch.step)
        else:
            finished = [clip_group(g, MAX_SEQ_LEN) for g in finished]
            await model.train(
                trajectory_groups=finished,
                config=art.TrainConfig(learning_rate=training_config["learning_rate"]),
            )
            wandb.log({"train/used_judged_groups": 0}, step=batch.step)

        if batch.step >= training_config["max_steps"]:
            break

    wandb.finish()

if __name__ == "__main__":
    asyncio.run(main())
