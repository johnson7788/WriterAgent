#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
更稳健的奖励函数（format_reward / search_reward）示例
- 结构软匹配：允许长度/顺序轻微不一致，按比例给分
- 文本填充：以“应填位置”为分母；长度阈值放宽；模板句降权而非直接判 0
- 检索奖励：只看来源存在、域名多样性、引用索引有效、（温和）工具一致性
- 平滑与保底：有任意实质填充即给小额保底，避免动不动 0 分

运行方式：
    python reward_demo.py
"""
import re
import json
import urllib.parse
from typing import List, Dict, Any, Optional, Tuple


# ========= 工具函数 =========
def _norm(s: Optional[str]) -> str:
    return str(s or "").strip().lower()


def _url_domain(u: str) -> str:
    try:
        return urllib.parse.urlparse(u).netloc.lower()
    except Exception:
        return ""


def _expected_text_paths(task: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """根据“输入大纲”，返回期望填写 text 的 (i, j) 路径：i=task 索引，j=items 索引。"""
    paths = []
    for i, node in enumerate(task or []):
        if (node or {}).get("type") == "content":
            items = (node.get("data") or {}).get("items", [])
            if isinstance(items, list):
                for j, it in enumerate(items):
                    if isinstance(it, dict):
                        paths.append((i, j))
    return paths


def _get_out_text_at(out_task: List[Dict[str, Any]], i: int, j: int) -> str:
    try:
        node = out_task[i]
        items = (node.get("data") or {}).get("items", [])
        if not isinstance(items, list) or j >= len(items):
            return ""
        return str((items[j] or {}).get("text", "") or "")
    except Exception:
        return ""


# ========= 结构软匹配 =========
def _structure_match_score(inp: List[Dict[str, Any]], outp: List[Dict[str, Any]]) -> float:
    """
    软评分（0~1）：
    - type 序列匹配（权重 0.4）
    - data.title 匹配（权重 0.2）
    - content 的 items 数量匹配（权重 0.2）
    - item.title 匹配比例（权重 0.2）
    任何一项缺失都按能匹配的部分给分，不做硬性 0。
    """
    try:
        if not isinstance(outp, list) or not isinstance(inp, list) or len(inp) == 0:
            return 0.0

        n = max(len(inp), len(outp))

        # 1) type 匹配
        type_hits = 0
        for k in range(min(len(inp), len(outp))):
            if inp[k].get("type") == outp[k].get("type"):
                type_hits += 1
        type_score = type_hits / n

        # 2) title 匹配
        title_total = 0
        title_hits = 0
        for k in range(min(len(inp), len(outp))):
            at, bt = inp[k].get("type"), outp[k].get("type")
            if at == bt and at in ("cover", "transition", "content"):
                a_title = _norm((inp[k].get("data") or {}).get("title"))
                b_title = _norm((outp[k].get("data") or {}).get("title"))
                if a_title:
                    title_total += 1
                    if a_title == b_title:
                        title_hits += 1
        title_score = (title_hits / title_total) if title_total else 1.0

        # 3) items 数量匹配 & 4) item.title 匹配
        items_total = 0
        items_hits = 0
        item_title_total = 0
        item_title_hits = 0
        for k in range(min(len(inp), len(outp))):
            if inp[k].get("type") == "content" and outp[k].get("type") == "content":
                aitems = (inp[k].get("data") or {}).get("items", [])
                bitems = (outp[k].get("data") or {}).get("items", [])
                if isinstance(aitems, list) and isinstance(bitems, list):
                    items_total += 1
                    if len(aitems) == len(bitems):
                        items_hits += 1
                    for ai, bi in zip(aitems, bitems):
                        at, bt = _norm((ai or {}).get("title")), _norm((bi or {}).get("title"))
                        if at:
                            item_title_total += 1
                            if at == bt:
                                item_title_hits += 1

        items_score = (items_hits / items_total) if items_total else 1.0
        item_title_score = (item_title_hits / item_title_total) if item_title_total else 1.0

        return (
            0.40 * type_score +
            0.20 * title_score +
            0.20 * items_score +
            0.20 * item_title_score
        )
    except Exception:
        return 0.0


# ========= 主体格式奖励 =========
def format_reward(inp_task: List[Dict[str, Any]], out_task: List[Dict[str, Any]]) -> float:
    """
    0~1：结构软匹配 + 文本填充 + 引用存在 + JSON 合法
    - 文本长度阈值：40
    - 模板句（以 "detailed content" 开头）降权但不直接判 0
    - 有任意实质填充给 0.05 保底
    """
    try:
        s_score = _structure_match_score(inp_task, out_task)  # 结构 50%

        paths = _expected_text_paths(inp_task)
        if not paths:
            try:
                json.dumps(out_task, ensure_ascii=False)
                json_ok = 1.0
            except Exception:
                json_ok = 0.0
            return 0.70 * s_score + 0.30 * json_ok

        filled = 0
        quality = 0
        for (i, j) in paths:
            txt = _get_out_text_at(out_task, i, j).strip()
            if txt:
                filled += 1
                ok_len = len(txt) >= 40
                not_template = not _norm(txt).startswith("detailed content")
                if ok_len and not_template:
                    quality += 1

        fill_rate = filled / len(paths)                  # 写了就有分
        qual_rate = (quality / filled) if filled else 0  # 写得像样的比例

        # 引用存在率（只看是否出现过 [n]）
        txts = [_get_out_text_at(out_task, i, j) for (i, j) in paths]
        cite_any = 1.0 if any(re.search(r"\[\d+\]", t or "") for t in txts) else 0.0

        try:
            json.dumps(out_task, ensure_ascii=False)
            json_ok = 1.0
        except Exception:
            json_ok = 0.0

        base = 0.50 * s_score + 0.30 * (0.5 * fill_rate + 0.5 * qual_rate) + 0.10 * cite_any + 0.10 * json_ok

        if filled > 0:
            base = max(base, 0.05)  # 保底

        return max(0.0, min(1.0, base))
    except Exception:
        return 0.0


# ========= 检索奖励 =========
def search_reward(out_task: List[Dict[str, Any]], sources: List[str],
                  tool_urls_seen: Optional[List[str]] = None) -> float:
    """
    0~1：来源存在 + 域名多样性 + 引用索引有效 + 工具一致性（温和）
    - 只要有来源就有分；没有来源才 0
    - 域名多样性：上限 3 个域名即可拉满
    - 工具一致性：0.9~1.0 的轻量乘子，避免一票否决
    """
    try:
        sources = [s for s in (sources or []) if isinstance(s, str) and s.startswith("http")]
        if not sources:
            return 0.0

        # 来源数量（上限 5）
        k = len(set(sources))
        src_score = min(k, 5) / 5.0  # 0~1 的比例

        # 域名多样性（上限 3）
        domains = {_url_domain(u) for u in sources if _url_domain(u)}
        diversity = min(len(domains), 3) / 3.0

        # 引用索引有效
        texts = []
        for node in out_task or []:
            if (node or {}).get("type") == "content":
                items = (node.get("data") or {}).get("items", [])
                for it in items or []:
                    t = (it or {}).get("text")
                    if t:
                        texts.append(str(t))
        total_cites = 0
        ok_cites = 0
        for t in texts:
            for m in re.finditer(r"\[(\d+)\]", t):
                total_cites += 1
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(sources):
                    ok_cites += 1
        cite_valid = (ok_cites / total_cites) if total_cites else 1.0  # 没写引用不扣分

        # 工具一致性（温和加成）
        tool_factor = 1.0
        if tool_urls_seen:
            tool_set = set(tool_urls_seen)
            if tool_set:
                in_tool = [u for u in sources if u in tool_set]
                ratio = len(in_tool) / len(sources)
                tool_factor = 0.9 + 0.1 * ratio  # 0.9~1.0

        score = (0.50 * src_score) + (0.30 * diversity) + (0.20 * cite_valid)
        return max(0.0, min(1.0, score * tool_factor))
    except Exception:
        return 0.0


# ========= 组合评估 =========
def combined_reward(inp_task: List[Dict[str, Any]], out_task: List[Dict[str, Any]],
                    sources: List[str], tool_urls_seen: Optional[List[str]] = None) -> Dict[str, float]:
    fr = format_reward(inp_task, out_task)
    sr = search_reward(out_task, sources, tool_urls_seen=tool_urls_seen)
    total = 0.5 * fr + 0.5 * sr
    return {"format_reward": fr, "search_reward": sr, "total_reward": total}


# ========= 简单测试 =========
def _build_sample_tasks():
    # 输入大纲（应填 4 个 text）
    inp_task = [
        {"type": "cover", "data": {"title": "2025年市场趋势与增长机会", "text": "A presentation"}},
        {"type": "content", "data": {"title": "人工智能技术应用", "items": [
            {"title": "部署AI辅助诊断系统", "text": ""},
            {"title": "开发智能推荐系统", "text": ""}
        ]}},
        {"type": "content", "data": {"title": "数字化转型加速", "items": [
            {"title": "推动设备加装智能传感器", "text": ""},
            {"title": "推广算力资源普惠服务", "text": ""}
        ]}},
        {"type": "end"}
    ]

    # 情形 A：较好输出（结构匹配、长度足、引用有效）
    out_task_good = [
        {"type": "cover", "data": {"title": "2025年市场趋势与增长机会", "text": "A presentation"}},
        {"type": "content", "data": {"title": "人工智能技术应用", "items": [
            {"title": "部署AI辅助诊断系统", "text":
                "多家三甲医院在影像科室上线 AI 读片辅助，平均报告出具时间缩短 20% 以上；"
                "在心脑血管筛查上，AI 提示敏感病灶的召回率提升到 0.92。试点单位配套建立责任闭环与可追溯审签体系。[1][2]"},
            {"title": "开发智能推荐系统", "text":
                "面向电商场景，通过向量检索与重排模型结合，点击率提升 8%~12%；"
                "冷启动阶段引入用户画像与内容相似度，减少新客无结果比例至 1% 以下。[1][3]"}
        ]}},
        {"type": "content", "data": {"title": "数字化转型加速", "items": [
            {"title": "推动设备加装智能传感器", "text":
                "在产线关键工位部署振动与温度传感器，结合阈值+异常检测模型，实现设备故障的小时级预警；"
                "以汽车零部件厂为例，不良率 3 个月内下降至 0.6%。[2]"},
            {"title": "推广算力资源普惠服务", "text":
                "采用集中 GPU 资源池与配额制度，训练作业排队时间由高峰期 36h 降至 6h 以内；"
                "团队通过 A/B 验证，离线特征刷新频率从周级提升到日级，转化率随之上涨 2%。[3]"}
        ]}},
        {"type": "end"}
    ]
    sources_good = [
        "https://example-hospital.org/ai-imaging-report",
        "https://example-factory.com/sensor-case",
        "https://example-commerce.com/rec-system"
    ]
    tool_urls_seen = [
        "https://example-hospital.org/ai-imaging-report",
        "https://example-commerce.com/rec-system",
        # 少一个也没关系，只是轻量乘子
    ]

    # 情形 B：部分填充（有字但短、引用缺失）
    out_task_partial = [
        {"type": "cover", "data": {"title": "2025年市场趋势与增长机会", "text": "A presentation"}},
        {"type": "content", "data": {"title": "人工智能技术应用", "items": [
            {"title": "部署AI辅助诊断系统", "text": "已在多家医院试点，效率提升明显。"},
            {"title": "开发智能推荐系统", "text": "通过召回与重排结合，CTR 有提升。"}
        ]}},
        {"type": "content", "data": {"title": "数字化转型加速", "items": [
            {"title": "推动设备加装智能传感器", "text": ""},
            {"title": "推广算力资源普惠服务", "text": "资源池统一调度，排队时间下降。"}
        ]}},
        {"type": "end"}
    ]
    sources_partial = ["https://example.com/post"]

    # 情形 C：结构有偏差（items 数不等），但不应被判 0
    out_task_mismatch = [
        {"type": "cover", "data": {"title": "2025年市场趋势与增长机会", "text": "A presentation"}},
        {"type": "content", "data": {"title": "人工智能技术应用", "items": [
            {"title": "部署AI辅助诊断系统", "text": "Detailed content about ...（模板句，降权）"},
        ]}},
        {"type": "content", "data": {"title": "数字化转型加速", "items": [
            {"title": "推动设备加装智能传感器", "text": "有一定效果。[1]"},
            {"title": "新增条目（不在输入里）", "text": "多写了一条，会被结构分扣掉。"}
        ]}},
        {"type": "end"}
    ]
    sources_mismatch = ["https://example-factory.com/sensor-case"]

    # 情形 D：完全没填 & 没来源（应为 0 或接近 0）
    out_task_empty = [
        {"type": "cover", "data": {"title": "2025年市场趋势与增长机会", "text": "A presentation"}},
        {"type": "content", "data": {"title": "人工智能技术应用", "items": [
            {"title": "部署AI辅助诊断系统", "text": ""},
            {"title": "开发智能推荐系统", "text": ""}
        ]}},
        {"type": "content", "data": {"title": "数字化转型加速", "items": [
            {"title": "推动设备加装智能传感器", "text": ""},
            {"title": "推广算力资源普惠服务", "text": ""}
        ]}},
        {"type": "end"}
    ]
    sources_empty: List[str] = []

    return {
        "A_good": (inp_task, out_task_good, sources_good, tool_urls_seen),
        "B_partial": (inp_task, out_task_partial, sources_partial, []),
        "C_mismatch": (inp_task, out_task_mismatch, sources_mismatch, []),
        "D_empty": (inp_task, out_task_empty, sources_empty, []),
    }


def _print_scores(tag: str, scores: Dict[str, float]):
    fr = scores["format_reward"]
    sr = scores["search_reward"]
    tr = scores["total_reward"]
    print(f"[{tag}] format_reward={fr:.3f} | search_reward={sr:.3f} | total={tr:.3f}")


if __name__ == "__main__":
    cases = _build_sample_tasks()
    for tag, (inp_task, out_task, sources, tool_urls_seen) in cases.items():
        scores = combined_reward(inp_task, out_task, sources, tool_urls_seen=tool_urls_seen)
        _print_scores(tag, scores)

    # 预期（非严格）：
    # - A_good：各项较高（~0.6-0.9）
    # - B_partial：中等但非 0（~0.2-0.5）
    # - C_mismatch：结构被扣，但只要有填充/来源仍应 > 0（~0.1-0.4）
    # - D_empty：接近 0（可能 0.0~0.05 之间）
