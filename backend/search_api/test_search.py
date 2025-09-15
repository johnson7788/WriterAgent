#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/4 10:28
# @File  : test_search.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 测试搜索APi的代码

import os
import unittest
import time
import json
import random
import string
import httpx
from httpx import AsyncClient

class SearchApiTestCase(unittest.IsolatedAsyncioTestCase):
    """
    Tests for the WeChat Search API
    - POST /api/search_keyword  (search mode -> hits with file_id)
    - POST /api/search_keyword  (by_id mode -> full content)
    """

    host = os.environ.get('host', 'http://127.0.0.1')
    port = os.environ.get('port', 10052)
    base_url = f"{host}:{port}"

    keyword = os.environ.get("TEST_KEYWORD", "吉利汽车")
    limit = int(os.environ.get("TEST_LIMIT", "3"))

    # ---------- 工具函数 ----------
    def _search_once(self, keyword=None, limit=None):
        """Sync helper to call /api/search_keyword in 'search' mode."""
        url = f"{self.base_url}/api/search_keyword"
        data = {
            "keyword": keyword or self.keyword,
            "limit": limit or self.limit
        }
        headers = {'content-type': 'application/json'}
        start_time = time.time()
        resp = httpx.post(url, json=data, headers=headers, timeout=None)
        took = time.time() - start_time
        print(f"[search] status={resp.status_code}, took={took:.2f}s, server={self.host}")
        self.assertEqual(resp.status_code, 200, "/api/search_keyword should return 200")
        return resp.json()

    async def _search_once_async(self, keyword=None, limit=None):
        """Async helper to call /api/search_keyword in 'search' mode."""
        url = f"{self.base_url}/api/search_keyword"
        data = {
            "keyword": keyword or self.keyword,
            "limit": limit or self.limit
        }
        headers = {'content-type': 'application/json'}
        start_time = time.time()
        async with AsyncClient() as client:
            resp = await client.post(url, json=data, headers=headers, timeout=None)
        took = time.time() - start_time
        print(f"[search-async] status={resp.status_code}, took={took:.2f}s, server={self.host}")
        self.assertEqual(resp.status_code, 200, "/api/search_keyword should return 200")
        return resp.json()

    async def _by_ids_async(self, file_ids, keyword="占位"):
        """Async helper to call /api/search_keyword in 'by_id' mode."""
        url = f"{self.base_url}/api/search_keyword"
        data = {
            "keyword": keyword,
            "file_ids": file_ids
        }
        headers = {'content-type': 'application/json'}
        start_time = time.time()
        async with AsyncClient() as client:
            resp = await client.post(url, json=data, headers=headers, timeout=None)
        took = time.time() - start_time
        print(f"[by_id] status={resp.status_code}, took={took:.2f}s, server={self.host}")
        self.assertEqual(resp.status_code, 200, "/api/search_keyword(by_id) should return 200")
        return resp.json()

    # ---------- 测试用例 ----------
    def test_search_keyword_hits(self):
        """测试搜索命中列表（含 file_id/snippet）"""
        response_data = self._search_once(keyword="吉利汽车")
        print(f"搜索的结果是: {response_data}")
        # 基本结构
        self.assertIn("articles", response_data)
        self.assertIn("keyword", response_data)
        self.assertIn("count", response_data)
        assert response_data["count"] >= 0, "搜索的count 应该大于等于0"
        if response_data["count"] == 0:
            self.skipTest("搜索结果为0，可能是外部站点限流或网络问题，跳过后续断言。")

        # 检查第一条结构
        art = response_data["articles"][0]
        for k in ["id", "file_id", "title", "snippet"]:
            self.assertIn(k, art, f"missing key '{k}' in article")
        self.assertTrue(isinstance(art["file_id"], str) and len(art["file_id"]) >= 8)

    async def test_by_ids_returns_full_content(self):
        """异步测试，并根测试根据文件id进行搜索，
        测试按 file_ids 批量返回完整正文,
        # 先搜吉利汽车，然后根据返回的结果的id，搜索包含吉利银河的结果
        """
        first_keyword = "吉利汽车"  # 宽泛的搜索
        second_keyword = "吉利银河" # 在搜索结果中继续搜索
        response_data = await self._search_once_async(keyword="吉利汽车")
        print(f"宽泛的搜索结果是: {response_data}")
        assert response_data["count"] >= 0, "搜索的count 应该大于等于0"
        file_ids = [a["file_id"] for a in response_data["articles"]]
        byid_response = await self._by_ids_async(file_ids=file_ids[:2], keyword="吉利银河")
        print(f"根据文件id继续搜索的结果是: {byid_response}")
        self.assertIn("mode", byid_response)
        self.assertEqual(byid_response["mode"], "by_id")
        self.assertIn("articles", byid_response)
        doc = byid_response["articles"][0]
        for k in ["file_id", "title", "content"]:
            self.assertIn(k, doc, f"missing key '{k}' in doc")
        self.assertIsInstance(doc["content"], str)
        self.assertGreater(len(doc["content"]), 0, "content 应该非空")

if __name__ == "__main__":
    unittest.main()
