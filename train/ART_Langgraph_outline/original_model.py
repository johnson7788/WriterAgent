#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 本地模型评测版（Ollama）：按 topic 搜索并生成 Markdown 大纲
# # 安装 & 启动略，确保本地能访问 11434 端口
# ollama pull qwen2.5:7b
# pip intall langchain_ollama

import os
import uuid
import asyncio
from typing import List, Optional
import dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel
from langchain_ollama import ChatOllama  # ✅ 关键：本地模型
from zai import ZhipuAiClient
import prompt

dotenv.load_dotenv()

# 业务搜索客户端（你原来的）
WebSearchClient = ZhipuAiClient(api_key=os.environ["ZHIPU_API_KEY"])

class WebSearchResult(BaseModel):
    url: str
    title: str
    snippet: str

def search_web(keyword: str) -> List[WebSearchResult]:
    resp = WebSearchClient.web_search.web_search(
        search_engine="search_std",
        search_query=keyword,
        count=4,
        search_recency_filter="noLimit",
        content_size="high",
    )
    return [
        WebSearchResult(url=item["url"], title=item["title"], snippet=item["content"])
        for item in resp["search_result"]
    ]

async def run_agent_eval(topic: str):
    final_outline: Optional[str] = None

    @tool
    def web_search_tool(query: str) -> List[dict]:
        """根据查询词进行网络搜索，返回结果列表。"""
        return [r.model_dump() for r in search_web(query)]

    @tool
    def return_final_outline_tool(outline: str, source_urls: List[str]) -> dict:
        """提交最终大纲以及引用来源 URL 列表。"""
        nonlocal final_outline
        final_outline = outline
        return {"outline": outline, "source_urls": source_urls}

    tools = [web_search_tool, return_final_outline_tool]

    # ✅ 用本地 Ollama 模型
    chat_model = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        # 如果你的 Ollama 不在默认端口，解开下一行并改地址
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.3,
    )

    agent = create_react_agent(chat_model, tools)
    res = await agent.ainvoke(
        {
            "messages": [
                SystemMessage(content=prompt.ROLLOUT_SYSTEM_PROMPT),
                HumanMessage(content=prompt.ROLLOUT_USER_PROMPT.format(topic=topic)),
            ]
        },
        config={"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 12},
    )

    print("====== 推理返回（含工具轨迹） ======")
    print(res)
    print("====== 最终大纲(如已提交) ======")
    if final_outline:
        print(final_outline)
    print("[EVAL] baseline local model finished.")

async def main():
    topic = os.getenv("OUTLINE_TOPIC") or "AIGC 在医疗影像的应用趋势"
    await run_agent_eval(topic)

if __name__ == "__main__":
    asyncio.run(main())
