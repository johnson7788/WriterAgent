#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/6/20 10:02
# @File  : tools.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :
import os
import httpx
from datetime import datetime
import random
import dotenv
from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool
import time
dotenv.load_dotenv()

def search_api(keyword: str, limit: int = 4):
    """
    调用搜索接口进行搜索
    Returns:
        results: list[dict], 每个元素的内容包括下面的4个字段
    'keyword' = {str} '类器官 结直肠癌 耐药机制'
    'count' = {int} 4
    'mode' = {str} 'search'
    'articles' = {list: 4} [{'file_id': '3f1fa5613fd19227', 'id': 'afd95dc1-7006-445d-827f-6bd891f1dc30', 'publish_time': '2022-12-27 17:41:17', 'score': 1.0, 'snippet': '文献\nID\n原名\n：\nMex3a marks drug-tolerant persister colorectal cancer cells that mediate relapse after chemotherapy\n译名：\nMex3...企业等。公司具备持续的创新能力，\n通过自主研发累计申请80余项核心专利与PCT专利，目前，已获得近50项授\n权。\n联系我们\n座机：400-990-8020\n企业邮箱：CXGJ@accibio.com\n网站：www.3dbudcare.com', 'title': '文献解读 I 肿瘤类器官模型助力挖掘新结直肠肿瘤细胞群化疗耐药新机制(下)', 'url': 'https://mp.weixin.qq.com/s?src=11&timestamp=1756956639&ver=6215&signature=RnUZANNR5sdS2TyxG*BzDg6Bp5M254GzGMbvA9vcdGsiLxMOf0F*77QeeHOwzuwocfCR80rhvA714gsXTH14pgdwW1wtGnv6eiyaVGef1yWtXb6-peV6kv2DkR1H5JOX&new=1'}, {'file_id': '180ce626b48991de', 'id': 'e3d12b48-872a-40af-aeff-0fa47cb73a30', 'publish_time': '2022-10-31 09:59:31', 'score': 1.0, 'snippet': '2022年8月天津医科大学肿瘤医院胸外科Zhentao Yu, Haiyang Zhang和Yi Ba（通讯作者）团队在Advanced Science期刊（影响因子17.521）上发表了一篇题为“Adipocyte-derived exosomal...养基试剂盒（Cat# K2O-M-CO）进行相关结直肠癌类器官模型的构建及扩增。我公司除了结直肠癌类器官培养基试剂盒...
    """
    SEARCH_TOOL_API = os.environ["SEARCH_TOOL_API"]
    url = f"{SEARCH_TOOL_API}/api/search_keyword"
    data = {
        "keyword": keyword,
        "limit": limit
    }
    headers = {'content-type': 'application/json'}
    start_time = time.time()
    resp = httpx.post(url, json=data, headers=headers, timeout=None, trust_env=False)
    took = time.time() - start_time
    print(f"[search] status={resp.status_code}, took={took:.2f}s")
    reponse_data = resp.json()
    return reponse_data


async def DocumentSearch(
        keyword: str,
        tool_context: ToolContext,
):
    """
    根据关键词搜索文档
    :param keyword: str, 搜索的相关文档的关键词
    :return: 返回每篇文档数据
    """
    agent_name = tool_context.agent_name
    print(f"Agent {agent_name} 正在调用工具：DocumentSearch: " + keyword)
    metadata = tool_context.state.get("metadata", {})
    print(f"调用工具：DocumentSearch时传入的metadata: {metadata}")
    # 从工具上下文中获取或初始化 references 字典
    # 现在 references 将存储更详细的文档信息
    references = tool_context.state.get("references", {})
    print(f"调用工具：DocumentSearch时传入的references: {references}")
    print("文档检索: " + keyword)

    start_time = time.time()
    results = search_api(keyword, limit=10)
    if not results or not results.get("articles"):
        return f"没有搜索到{keyword}相关的文章"

    articles = results["articles"]
    end_time = time.time()
    print(f"关键词{keyword}相关的文章已经获取完毕，获取到{len(articles)}篇, 耗时{end_time - start_time}秒")

    # 遍历新搜索到的文章，并更新 references 字典
    # 确保 references 字典中包含所有所需的信息
    for article_dict in articles:
        file_id = article_dict["file_id"]
        # 如果该文档ID不在 references 中，则添加它
        if file_id not in references:
            # 使用当前 references 的长度作为新文档的索引值
            idx_val = len(references) + 1
            article_url = article_dict["url"]
            article_title = article_dict["title"]
            if not article_url:
                article_url = "https://www.bing.com/search?q=" + article_title
            references[file_id] = {
                "idx_val": idx_val,
                "file_id": file_id,
                "title": article_dict["title"],
                "publish_time": article_dict["publish_time"],
                "snippet": article_dict["snippet"],
                "url": article_url
            }

    # 准备返回给用户的新文章内容列表
    articles_content = []
    for article_dict in articles:
        file_id = article_dict["file_id"]
        # 从更新后的 references 字典中获取完整的文档信息
        reference_info = references.get(file_id, {})

        # 提取所需的信息来格式化输出字符串
        idx_val = reference_info.get("idx_val", "N/A")
        title = reference_info.get("title", "N/A")
        publish_time = reference_info.get("publish_time", "N/A")
        snippet_content = reference_info.get("snippet", "N/A")

        articles_content.append(
            f"{idx_val}. 标题：{title}\n时间: {publish_time}\n内容: {snippet_content}\n\n"
        )

    # 重新设置 tool_context 中的 references
    tool_context.state["references"] = references

    return articles_content

if __name__ == '__main__':
    result = DocumentSearch(keyword="电动汽车")
    print(result)