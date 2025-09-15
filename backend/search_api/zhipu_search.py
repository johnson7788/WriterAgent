#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/4 14:24
# @File  : zhipu_search.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 智谱搜索，GLM的搜索
import os
import dotenv
import logging
from pydantic import BaseModel, Field
from cache_utils import cache_decorator
from zai import ZhipuAiClient
logger = logging.getLogger(__name__)

dotenv.load_dotenv()
WebSearchClient = ZhipuAiClient(api_key=os.environ["ZHIPU_API_KEY"])

@cache_decorator
def zhipu_search_web(keyword: str, number=10) -> list[dict]:
    """
    智谱的网络搜索
    Returns:

    """
    logger.info(f"[zhipu_search] keyword={keyword}")
    response = WebSearchClient.web_search.web_search(
        search_engine="search_std",
        search_query=keyword,
        count=number,  # 返回结果的条数，范围1-50，默认10
        search_recency_filter="noLimit",  # 搜索指定日期范围内的内容
        content_size="high"  # 控制网页摘要的字数，默认medium
    )
    search_result = response.search_result
    data = []
    for item in search_result:
        one_dump_data = item.model_dump()
        one_data = {
            "title": one_dump_data["title"],
            "publish_time": one_dump_data["publish_date"],
            "url": one_dump_data["link"],
            "content": one_dump_data["content"]
        }
        data.append(one_data)
    logger.info(f"[zhipu_search] result={data}")
    if not data:
        return False, []
    return True, data

if __name__ == '__main__':
    print(zhipu_search_web("RAG"))