#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/10 15:44
# @File  : utils.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 工具函数


import json
import re
from typing import Any, Dict, Optional, Iterable, List
from datetime import datetime
from urllib.parse import urlparse

def stringify_references(references: Dict[str, Any]) -> Optional[str]:
    """
    从读取 references，并序列化为“论文参考文献式”的长文本（换行拼接）。
    - 若为 str：去空格后直接返回；
    - 若为 list/dict：转成按序编号的参考文献条目文本；
    - 若为空或不存在：返回 None。

    支持的条目字段：
      - title: 必选（若缺失则用“（无标题）”）
      - author: 必选（若缺失则用“（未知）”）
      - journal: 可选（若缺失则不显示期刊名）
      - docdoi: 可选（若缺失则不显示 DOI）
      - publish_time: 可选（尝试标准化为 YYYY-MM-DD）
      - url: 可选（用于输出“可获取于”并解析来源域名）
      - idx_val: 可选（若存在，用于排序；否则保持原有顺序）
    """
    print(f"开始解析:references:\n{references}")
    if references is None:
        return None
    if isinstance(references, str):
        s = references.strip()
        return s if s else None

    # 将 references 统一摊平成条目列表
    items: List[Dict[str, Any]] = []
    if isinstance(references, dict):
        for k, v in references.items():
            if isinstance(v, dict):
                entry = dict(v)  # 浅拷贝，避免修改原对象
                entry.setdefault("file_id", k)
                items.append(entry)
            else:
                # 容错：非 dict 的值也尽量转成条目
                items.append({"file_id": k, "title": str(v)})
    elif isinstance(references, list):
        for v in references:
            if isinstance(v, dict):
                items.append(v)
            else:
                items.append({"title": str(v)})
    else:
        # 其他类型兜底：尽量转字符串
        try:
            return json.dumps(references, ensure_ascii=False)
        except Exception:
            return str(references)

    if not items:
        return None

    # 是否存在 idx_val；有则按 idx_val 排序，否则保留原顺序
    has_idx = any("idx_val" in it and it.get("idx_val") is not None for it in items)

    def _idx_safe(val):
        try:
            return int(val)
        except Exception:
            return float("inf")

    if has_idx:
        items.sort(key=lambda d: (_idx_safe(d.get("idx_val")), str(d.get("publish_time") or "")))

    today = datetime.now().strftime("%Y-%m-%d")

    def _norm_date(s: Any) -> Optional[str]:
        """尽量把日期标准化为 YYYY-MM-DD；识别 ISO/常见格式，失败则返回 None。"""
        if not s:
            return None
        if isinstance(s, (int, float)):
            try:
                # 兼容时间戳（秒）
                return datetime.fromtimestamp(float(s)).strftime("%Y-%m-%d")
            except Exception:
                return None
        s = str(s).strip()
        if not s:
            return None
        # 尝试匹配 YYYY-MM-DD
        m = re.match(r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})", s)
        if m:
            y, mo, d = m.groups()
            try:
                return datetime(int(y), int(mo), int(d)).strftime("%Y-%m-%d")
            except Exception:
                pass
        # 尝试匹配 YYYY/MM/DD
        m = re.match(r"^\s*(\d{4})/(\d{1,2})/(\d{1,2})", s)
        if m:
            y, mo, d = m.groups()
            try:
                return datetime(int(y), int(mo), int(d)).strftime("%Y-%m-%d")
            except Exception:
                pass
        # 尝试从 ISO8601 中抽取日期
        m = re.match(r"^\s*(\d{4})-(\d{2})-(\d{2})[T\s]", s)
        if m:
            y, mo, d = m.groups()
            try:
                return datetime(int(y), int(mo), int(d)).strftime("%Y-%m-%d")
            except Exception:
                pass
        return None

    lines = []
    for i, it in enumerate(items, 1):
        title = (it.get("title") or "").strip()
        # 清理标题中的多余空格：将连续的空白字符（空格、制表符、换行符等）替换为单个空格
        title = re.sub(r'\s+', ' ', title)
        date_str = _norm_date(it.get("publish_time"))
        author = (it.get("author") or "").strip() or ""
        journal = (it.get("journal") or "").strip() or ""
        journal = re.sub(r'\s+', ' ', journal)
        doi = (it.get("docdoi") or "").strip() or ""
        # year = (it.get("publish_time") or "").strip() or ""
        src = None
        # if url:
        #     try:
        #         netloc = urlparse(url).netloc
        #         # 去掉常见的 www 前缀
        #         src = netloc[4:] if netloc.startswith("www.") else netloc
        #     except Exception:
        #         src = None

        # 参考文献行（AMA格式）
        # 结构：[序号] 作者姓名. 文章标题. 期刊名. 年份;卷(期):页码. doi:xxx
        
        year = ""
        if date_str:
            year = date_str.split("-")[0]  # 从日期中提取年份
        
        # 构建AMA格式引用
        ama_parts = []
        
        # [序号] 作者姓名
        if author:
            # 处理作者数量限制：超过6位时只显示前6位并添加et al.
            authors_list = [a.strip() for a in author.split(',') if a.strip()]
            if len(authors_list) > 6:
                limited_authors = ', '.join(authors_list[:6]) + ', et al'
            else:
                limited_authors = author
            ama_parts.append(f"[{i}] {limited_authors}.")
        else:
            ama_parts.append(f"[{i}]")
        
        # 文章标题
        ama_parts.append(f"{title}.")
        
        # 期刊名
        if journal:
            ama_parts.append(f"{journal}.")
        
        # 年份;卷(期):页码
        pub_info = []
        if year:
            pub_info.append(year)
        
        # vol_issue_pages = []
        # if volume:
        #     if issue:
        #         vol_issue_pages.append(f"{volume}({issue})")
        #     else:
        #         vol_issue_pages.append(volume)
        # elif issue:
        #     vol_issue_pages.append(f"({issue})")
        
        # if pages:
        #     vol_issue_pages.append(f":{pages}")
        
        # if vol_issue_pages:
        #     if year:
        #         pub_info.append(";" + "".join(vol_issue_pages))
        #     else:
        #         pub_info.append("".join(vol_issue_pages))
        
        if pub_info:
            ama_parts.append("".join(pub_info) + ".")
        
        # DOI信息
        if doi:
            if not doi.startswith("doi:"):
                doi = f"doi:{doi}"
            ama_parts.append(doi)
        
        line = " ".join(ama_parts)
        # 清理拼接后数据中的多个空格：将连续的空格替换为单个空格
        line = re.sub(r' +', ' ', line).strip()
        lines.append(line)

    # 以换行拼接成一个长文本
    result = "\n# 参考文献:\n\n" + "\n\n".join(lines).strip()
    return result if result else None

if __name__ == '__main__':
    references ={
            '01a48fbf946f66a7': {
            'file_id': '01a48fbf946f66a7', 
            'idx_val': 1, 
            'publish_time': '2023-02-01', 
            'title': 'Moving    towards a personalized approach in chronic lymphocytic leukemia. Seminars in    cancer biology', 
            'url': 'https://example.com/paper.pdf',
            'author': 'Eugene R Przespolewski, Jeffrey Baron, Farshid Kashef, Kai Fu, Sheila N Jani Sait, Francisco Hernandez-Ilizaliturri, James Thompson',
            'journal': 'Journal of the National Comprehensive Cancer Network : JNCCN',
            'docdoi': '10.6004/jnccn.2022.7069'
        }
    }
    result = stringify_references(references)
    print(result)
