#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/9/3 13:15
# @File  : convert_outline.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :
"""Markdown → JSON converter for hierarchical outlines (Chinese-friendly).
Features:
- Parses headings (#..######) and extracts numeric IDs (e.g., 1, 1.1, 2.2.1).
- Builds nested sections/subsections.
- Treats a level-2 heading exactly named '摘要' as a special abstract block:
  - bullet items under 摘要 are interpreted as key: value pairs
  - '关键词' is split into a list using Chinese/Western separators
  - keys like '方法（可选）' will have the parenthetical moved into the value
- Bullets under non-摘要 headings are collected into '要点'
- Prunes empty 'subsections' arrays
- CLI usage to read from a file or stdin and write to a file or stdout
"""
import argparse
import json
import re
import sys
from typing import Dict, Any, List, Optional, Tuple
from copy import deepcopy

SEP_KV = re.compile(r"[：:]", re.U)  # Chinese/Western colon
ID_RE = re.compile(r"^(\d+(?:\.\d+)*)(?:[\.．、]?\s*)(.*)$", re.U)
HDR_RE = re.compile(r"^(#{1,6})\s+(.*)\s*$", re.U)
BULLET_RE = re.compile(r"^\s*-\s+(.*)$", re.U)
PAREN_RE = re.compile(r"（[^）]*）", re.U)

# NEW: 去 HTML 标签 + 反转义 \[ \] \( \)
TAG_RE = re.compile(r"<[^>]+>", re.U)
ESCAPED_BRACKET_RE = re.compile(r"\\([\[\]\(\)])", re.U)

def normalize_inline(s: str) -> str:
    """
    规范化行内文本：
    1) 去除所有 HTML 标签，保留其内部文字；
    2) 将 \[ \] \( \) 反转义为 [ ] ( )；
    3) 合并多余空白。
    """
    s = TAG_RE.sub("", s)
    s = ESCAPED_BRACKET_RE.sub(lambda m: m.group(1), s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_keywords(val: str) -> List[str]:
    """Split Chinese or Western punctuation commonly used in keyword lists."""
    parts = re.split(r"[；;、，,\s]+", val.strip())
    return [p for p in parts if p]

def clean_key_parenthetical(k: str, v: str) -> (str, str):
    """
    If key contains a parenthetical like '方法（可选）',
    move it into the value and keep a cleaner key '方法'.
    """
    m = PAREN_RE.search(k)
    if m:
        paren = m.group(0)
        k2 = PAREN_RE.sub("", k).strip()
        v2 = (paren + (v if v else "")).strip()
        return (k2 or k, v2)
    return (k, v)

def parse_abstract_bullet(item: str, abstract: Dict[str, Any]) -> None:
    """Parse one bullet line under 摘要 into key/value or list."""
    if SEP_KV.search(item):
        k, v = SEP_KV.split(item, 1)
        k, v = k.strip(), v.strip()
        k, v = clean_key_parenthetical(k, v)
        if "关键词" in k:
            abstract["关键词"] = split_keywords(v)
        else:
            abstract[k] = v
    else:
        abstract.setdefault("其他", []).append(item)

def attach_point_to_current(stack: Dict[int, Dict[str, Any]], point: str) -> None:
    """Attach a bullet point to the deepest current node in the stack."""
    for lvl in range(6, 1, -1):
        node = stack.get(lvl)
        if node is not None:
            node.setdefault("要点", []).append(point)
            return

def prune_empty(node: Dict[str, Any]) -> Dict[str, Any]:
    """Remove empty subsections lists recursively."""
    subs = node.get("subsections")
    if isinstance(subs, list):
        node["subsections"] = [prune_empty(child) for child in subs if child]
        if not node["subsections"]:
            node.pop("subsections", None)
    return node

def parse_outline_markdown_to_json(md: str) -> Dict[str, Any]:
    """
    Core parser. Returns a dictionary with:
      - title: str
      - abstract: dict (optional)
      - sections: list of section nodes
    """
    doc: Dict[str, Any] = {"title": None, "sections": []}
    stack: Dict[int, Optional[Dict[str, Any]]] = {}  # heading level -> node
    in_abstract = False

    lines = md.splitlines()
    for raw in lines:
        # 先行级别的标准化，兼容 <a> 包裹与反斜杠转义的情况
        line = normalize_inline(raw.rstrip("\n"))
        if not line:
            continue

        # Heading?
        m_hdr = HDR_RE.match(line)
        if m_hdr:
            level = len(m_hdr.group(1))
            text = m_hdr.group(2).strip()

            # Title (# ...)
            if level == 1:
                doc["title"] = text
                in_abstract = False
                stack.clear()
                continue

            # 摘要 special-case (## 摘要)
            if level == 2 and ID_RE.match(text) is None and text == "Abstract" or text == "摘要":
                in_abstract = True
                doc["abstract"] = {}
                # Reset deeper stack levels
                stack = {k: v for k, v in stack.items() if k < 2}
                continue
            else:
                in_abstract = False

            node: Dict[str, Any] = {"title": None, "subsections": []}
            m_id = ID_RE.match(text)
            if m_id:
                node_id, title_text = m_id.group(1), m_id.group(2).strip()
                node["id"] = node_id
                node["title"] = title_text
            else:
                node["title"] = text

            # Place node into tree according to heading level
            stack[level] = node
            # Clear deeper levels
            for deeper in range(level + 1, 7):
                stack.pop(deeper, None)

            if level == 2:
                doc["sections"].append(node)
            else:
                parent = stack.get(level - 1)
                if parent is None:
                    # Fallback: attach to last top-level section if possible
                    parent = doc["sections"][-1] if doc.get("sections") else None
                if parent is not None:
                    parent.setdefault("subsections", []).append(node)
            continue

        # Bullet?
        m_bul = BULLET_RE.match(line)
        if m_bul:
            content = m_bul.group(1).strip()
            if in_abstract:
                parse_abstract_bullet(content, doc.setdefault("abstract", {}))
            else:
                attach_point_to_current(stack, content)
            continue

        # Other free text lines are ignored in this outline style

    # Prune empty subsections recursively
    sections = [prune_empty(s) for s in doc["sections"]]
    references, sections = extract_references_and_clean(sections)
    doc["sections"] = sections
    return doc, references


def extract_references_and_clean(sections: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    提取每章(顶层 id)的小节标题中的所有引用，并清理原始 sections 中的小节标题。
    返回: (references_by_chapter, cleaned_sections)
      - references_by_chapter: [{"id": "<chapter_id>",
                                 "references": [{"label": str, "ref_id": str}, ...]}, ...]
      - cleaned_sections: 清除了引用标记后的 sections 深拷贝
    """
    # 兼容形如 ([文本][数字]) 的引用（经过 normalize_inline 后，md_text2 里的 \[ \] 已反转义）
    REF_PATTERN = re.compile(r"\(\[([^\[\]]+)\]\[(\d+)\]\)")

    data = deepcopy(sections)
    refs_out: List[Dict] = []

    for chapter in data:
        chap_id = chapter.get("id")
        collected: List[Tuple[str, str]] = []
        seen = set()

        for sub in chapter.get("subsections", []) or []:
            title = sub.get("title", "") or ""

            # 1) 收集引用
            for m in REF_PATTERN.finditer(title):
                label, ref_id = m.group(1), m.group(2)
                key = (label, ref_id)
                if key not in seen:
                    seen.add(key)
                    collected.append(key)

            # 2) 清理标题中的引用 与 拖尾分隔
            cleaned = REF_PATTERN.sub("", title)
            cleaned = re.sub(r"[\s,，、;；]+$", "", cleaned)   # 去尾部分隔符/空白
            cleaned = re.sub(r"\s{2,}", " ", cleaned)         # 合并多余空格
            sub["title"] = cleaned.strip()

        refs_out.append({
            "id": str(chap_id) if chap_id is not None else "",
            "references": [{"label": l, "ref_id": r} for (l, r) in collected]
        })

    return refs_out, data


def main():
    # ---- 内置示例 ----
    md_text = """我将为您搜索非小细胞肺癌相关的研究进展文献，然后生成详细的大纲。我将为您搜索非小细胞肺癌相关的研究进展文献，然后生成详细的大纲。基于检索到的文献资料，我现在为您生成非小细胞肺癌研究进展的严谨生物医学综述大纲：

# 非小细胞肺癌精准治疗研究进展
## 摘要
- 背景：非小细胞肺癌占所有肺癌病例的85%，是全球癌症死亡率首位
- 目的：综述非小细胞肺癌精准治疗的最新进展与转化应用
- 方法：系统检索近5年权威指南、临床试验与基础研究证据
- 主要发现：免疫治疗、靶向治疗、ADC药物联合策略取得突破性进展
- 结论与展望：从分子分型到个体化治疗的精准医学范式正在重塑治疗格局
- 关键词：非小细胞肺癌；免疫治疗；靶向治疗；分子分型；精准医学；临床试验
## 1. 引言与疾病负担
### 1.1 流行病学/流行现状与分型 ([新民晚报][2]),([JAMA子刊][4])
"""
    md_text2 = """# 血小板及其衍生物在糖尿病足溃疡治疗中的作用及机制研究进展
## 摘要
- 背景：糖尿病足溃疡致残致死率高，治疗亟需创新
- 目的：综述血小板及其衍生物于糖尿病足溃疡修复进展与机制
- 方法（可选）：系统检索近十年指南、综述与原始文献
- 主要发现：血小板源物对溃疡修复、多通路调控具积极作用
- 结论与展望：需强化标准化、机制转化及个体化临床落地
- 关键词：血小板；血小板衍生物；糖尿病足溃疡；生物材料；修复机制；个体化治疗
## 1. 引言与疾病/问题负担
### 1.1 流行病学/流行现状与分型 <a target="_self" rel="noopener noreferrer nofollow" class="text-blue-600 hover:text-blue-800 transition-colors cursor-pointer font-semibold" href="#ref-3" data-author="Interpretation" data-index="3" data-reference-link="true">(\[Interpretation\]\[3\])</a>,<a target="_self" rel="noopener noreferrer nofollow" class="text-blue-600 hover:text-blue-800 transition-colors cursor-pointer font-semibold" href="#ref-77" data-author="Progress" data-index="77" data-reference-link="true">(\[Progress\]\[77\])</a>
- 糖尿病发病率逐年升高，足溃疡并发症比例高<a target="_self" rel="noopener noreferrer nofollow" class="text-blue-600 hover:text-blue-800 transition-colors cursor-pointer font-semibold" href="#ref-3" data-author="Interpretation" data-index="3" data-reference-link="true">(\[Interpretation\]\[3\])</a>
### 1.2 现有标准方案/路径与疗效边界 <a target="_self" rel="noopener noreferrer nofollow" class="text-blue-600 hover:text-blue-800 transition-colors cursor-pointer font-semibold" href="#ref-3" data-author="Interpretation" data-index="3" data-reference-link="true">(\[Interpretation\]\[3\])</a>,<a target="_self" rel="noopener noreferrer nofollow" class="text-blue-600 hover:text-blue-800 transition-colors cursor-pointer font-semibold" href="#ref-5" data-author="Continuous" data-index="5" data-reference-link="true">(\[Continuous\]\[5\])</a>
## 9. 结语
- 血小板及其衍生物在糖尿病足溃疡桥接再生医学与临床
## 10. 附录
- 术语/缩写表；图表清单；补充方法
"""
    for name, md in [("md_text", md_text), ("md_text2", md_text2)]:
        data, references = parse_outline_markdown_to_json(md)
        print(f"===== {name} data =====")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        print(f"===== {name} references =====")
        print(json.dumps(references, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
