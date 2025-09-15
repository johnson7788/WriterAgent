#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/8/27 11:41
# @File  : model_test.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

# -*- coding: utf-8 -*-
"""
用训练过的同名模型跑一次 ReAct 推理测试（同样走 LangGraph tools）
注意：
- 若你在同一后端进程中连续训练->测试，后端已加载最近的 LoRA。
- 若在新进程测试，请保持相同的 model.name / project，并连接到相同后端（本地或 SkyPilot）。
"""

import os
import uuid
import asyncio
from textwrap import dedent
from typing import List
import dotenv
import art
from art.langgraph import init_chat_model, wrap_rollout
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel
from zai import ZhipuAiClient
dotenv.load_dotenv()
# ---------- 与训练保持一致 ----------
NAME = os.getenv("ART_NAME", "web-search")
MODEL_NAME = os.getenv("ART_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROJECT_NAME = os.getenv("ART_PROJECT", "web-search-agent-training")
USE_LOCAL_BACKEND = os.getenv("ART_BACKEND", "local").lower() == "local"
WebSearchClient = ZhipuAiClient(api_key=os.environ["ZHIPU_API_KEY"])

# ---------- 业务工具（与训练相同或真实实现）----------
class WebSearchResult(BaseModel):
    url: str
    title: str
    snippet: str
def search_web(keyword: str) -> List[WebSearchResult]:
    """
    真实的网络搜索函数
    """
    response = WebSearchClient.web_search.web_search(
        search_engine="search_std",
        search_query=keyword,
        count=4,  # 返回结果的条数，范围1-50，默认10
        search_recency_filter="noLimit",  # 搜索指定日期范围内的内容
        content_size="high"  # 控制网页摘要的字数，默认medium
    )
    return [
        WebSearchResult(
            url=item['url'],
            title=item['title'],
            snippet=item['content']
        ) for item in response['search_result']
    ]


async def run_agent_test(model: art.Model):
    system_prompt = dedent('''
    You are a web search agent. Use tools to find information on the web.
    When done, provide a concise answer.
    ''')

    @tool
    def web_search_tool(query: str) -> List[dict]:
        """Search the web with a query."""
        results = search_web(query)
        return [r.model_dump() for r in results]

    tools = [web_search_tool]

    # 用 ART 的 init_chat_model 获取可用的聊天模型（后端会加载最近训练好的 LoRA）
    chat_model = init_chat_model(model, temperature=0.7)
    agent = create_react_agent(chat_model, tools)

    res = await agent.ainvoke(
        {"messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Who is the CFO of Tesla?"),
        ]},
        config={"configurable": {"thread_id": str(uuid.uuid4())},
                "recursion_limit": 10},
    )
    print(f"测试的输出结果: {res}")
    print("[TEST] agent finished. See backend logs / tracing for details.")


async def main():
    # 连接与注册后端
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

    # 关键：用 wrap_rollout 包装推理函数，确保 ART 上下文正确设置
    wrapped_test_func = wrap_rollout(model, run_agent_test)
    await wrapped_test_func(model)

if __name__ == "__main__":
    asyncio.run(main())
