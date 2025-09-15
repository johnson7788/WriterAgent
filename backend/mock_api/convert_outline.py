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
from typing import Dict, Any, List, Optional

SEP_KV = re.compile(r"[：:]", re.U)  # Chinese/Western colon
ID_RE = re.compile(r"^(\d+(?:\.\d+)*)(?:[\.．、]?\s*)(.*)$", re.U)
HDR_RE = re.compile(r"^(#{1,6})\s+(.*)\s*$", re.U)
BULLET_RE = re.compile(r"^\s*-\s+(.*)$", re.U)
PAREN_RE = re.compile(r"（[^）]*）", re.U)

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
        line = raw.rstrip("\n")

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
            if level == 2 and ID_RE.match(text) is None and text == "摘要":
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
    doc["sections"] = [prune_empty(s) for s in doc["sections"]]
    return doc

def main():
    md_text = """# 类器官技术在结直肠癌耐药机制研究与个体化治疗中的应用前景

## 摘要
- 背景：结直肠癌（CRC）负担与标准治疗现状，耐药问题的临床意义。
- 目的：阐述类器官（organoid）在解析耐药机制与推进个体化治疗中的价值与前景。
- 方法（可选）：检索范围、纳排标准与证据分级简述。
- 主要发现：类器官的建系、共培养与多组学整合在化疗/靶向/免疫耐药研究与用药预测中的核心进展。
- 结论与展望：转化路径、标准化挑战与临床落地前景。
- 关键词：结直肠癌；类器官；耐药机制；个体化治疗；肿瘤微环境；药物敏感性测试

## 1. 引言与疾病负担
### 1.1 结直肠癌流行病学与分子分型（CIN、MSI-H/dMMR、CIMP 等）
### 1.2 标准治疗路径：手术、放化疗（5-FU/奥沙利铂/伊立替康）、抗 EGFR/VEGF、BRAF/HER2 靶向、免疫治疗（PD-1）
### 1.3 耐药的类型与临床难题：原发/继发、系统性/部位特异、药物耐受细胞（DTPs）
### 1.4 研究模型需求：体外 2D、类器官、PDX 的互补性与证据等级

## 2. 类器官技术概述
### 2.1 定义与发展脉络：成人干细胞来源（ASCs）与 iPSC/ESC 衍生类器官
### 2.2 建系流程与培养条件
#### 2.2.1 样本来源：手术标本、穿刺/内镜活检、循环肿瘤细胞/腹水；正常/肿瘤配对
#### 2.2.2 基质与培养基：基底膜替代物、Wnt/R-spondin/EGF/Noggin 等因子
#### 2.2.3 扩增、冻存与传代；时间线与成功率
### 2.3 质量控制与表征
#### 2.3.1 形态学、STR 指纹、WES/RNA-seq、拷贝数/突变保真度、MSI/MMR、IHC/IF、支原体检测
#### 2.3.2 药敏终点与重复性：IC50、AUC、DSS；重复性与批间差
### 2.4 与其他模型比较：代表性、通量、成本、微环境保真度、伦理与可及性
### 2.5 类器官生物样本库与数据共享：标准化 SOP、元数据、可重复性

## 3. 结直肠癌耐药的分子与细胞机制
### 3.1 化疗耐药：5-FU/奥沙利铂/伊立替康相关通路（DNA 修复、药物转运、代谢重编程、凋亡/自噬）
### 3.2 抗 EGFR 与 MAPK 通路耐药：KRAS/NRAS/BRAF 突变、EGFR ECD 突变、HER2/MET/FGFR 扩增、MAPK 旁路再激活
### 3.3 PI3K/AKT/mTOR、TGF-β、WNT/β-catenin、NOTCH 等通路交互
### 3.4 肿瘤干细胞（CSC）与 EMT：可塑性、表观遗传调控与药物耐受
### 3.5 肿瘤微环境（TME）：CAF、免疫细胞、内皮细胞与细胞外基质；低氧与剪切力
### 3.6 DNA 修复缺陷与 MSI 状态对疗效的影响；ctDNA 动态与克隆演化
### 3.7 微生物组影响：如梭杆菌（F. nucleatum）介导化疗耐药的证据

## 4. 类器官在耐药机制研究中的应用
### 4.1 建立耐药类器官模型：长期药物暴露、脉冲诱导、克隆追踪
### 4.2 基因编辑与功能验证：CRISPR 敲除/敲入、突变回补、合成致死筛选
### 4.3 多组学整合：单细胞转录组/ATAC、蛋白组/磷酸化组、代谢组、空间组学
### 4.4 共培养与微环境重建
#### 4.4.1 类器官-CAF/免疫细胞（T/NK/Mφ）/内皮细胞/微生物共培养
#### 4.4.2 空气-液体界面（ALI）、微流控器官芯片（shear/浓度梯度）
### 4.5 特定耐药场景案例
#### 4.5.1 抗 EGFR 继发 RAS 突变与旁路放大
#### 4.5.2 BRAF V600E 联合方案耐药重编程
#### 4.5.3 MSI-H 对免疫治疗反应的异质性与免疫逃逸
#### 4.5.4 放疗增敏/耐受的类器官评估
### 4.6 药物组合优化：高通量组合筛选、药物相互作用（Bliss/HSA/SynergyFinder）

## 5. 类器官在个体化治疗中的转化应用
### 5.1 ex vivo 药敏测试工作流
#### 5.1.1 从患者活检到报告：时间线、样本量要求、可行性与失败补救
#### 5.1.2 读出指标与临床阈值：反应分类、ROC/AUC 关联
### 5.2 与临床疗效的相关性
#### 5.2.1 抗 EGFR、FOLFOX/FOLFIRI、BRAF/HER2 靶向、KRAS G12C 抑制剂、免疫治疗的前瞻/回顾性证据
#### 5.2.2 “共同临床试验”（co-clinical）与 n-of-1 决策支持
### 5.3 伴随诊断与分层策略
#### 5.3.1 基因突变/拷贝数/表达谱 × 类器官药敏的整合评分
#### 5.3.2 ctDNA 动态监测与重复建系
### 5.4 医疗流程与经济学
#### 5.4.1 报告格式与 MDT 融合
#### 5.4.2 医保支付与成本-效果
### 5.5 真实世界落地案例与教训

## 6. 计算与数据科学助力
### 6.1 多组学与药敏数据的特征工程与建模（Elastic Net、随机森林、XGBoost、深度学习）
### 6.2 联邦学习与隐私保护、模型泛化与外部验证
### 6.3 数字孪生与“虚拟试验”：个体层面疗效模拟与方案优化
### 6.4 药物组合的算法优化与适应性试验设计

## 7. 局限性与挑战
### 7.1 技术与生物学局限：建立成功率、选择偏倚（MSI/MSS、转移部位）、培养基质批次差、缺乏血管/免疫成分
### 7.2 标准化与质量体系：SOP、QA/QC 指标、跨中心一致性、比对材料与环评
### 7.3 法规伦理：残余临床用途、数据共享与知情同意、生物安全与病原风险
### 7.4 可扩展性与成本：通量、自动化、周转时间与临床窗口匹配
### 7.5 结果解释与报告可读性：临床可行动阈值与不确定性沟通

## 8. 未来方向与前景
### 8.1 类器官-芯片与血管化/免疫化/神经化“组装体”（assembloids）
### 8.2 可替代动物源基质（合成水凝胶）与 GMP 级材料
### 8.3 与放疗/介入/电场治疗/放免联合的 ex vivo 评估
### 8.4 微生物组精准干预（噬菌体/益生策略）与耐药逆转
### 8.5 快速建系与“床旁”平台：自动化、微型高通量与报告标准
### 8.6 临床试验生态：适应性篮式/平台试验、真实世界证据整合

## 9. 结语
- 类器官在 CRC 耐药解析与个体化决策中的“桥梁”角色
- 从证据累积到临床标准化落地的关键里程碑与协同需求

## 10. 附录"""

    data = parse_outline_markdown_to_json(md_text)
    json.dump(data, sys.stdout, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
