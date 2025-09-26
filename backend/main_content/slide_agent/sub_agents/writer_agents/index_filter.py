import re
from typing import Dict, List, Tuple, Set, Optional

BRACKET_PATTERN = re.compile(r'\[\s*([0-9,\-\s]+?)\s*\]')

def _parse_bracket_content(content: str) -> List[int]:
    parts = [p.strip() for p in content.split(',') if p.strip()]
    nums: List[int] = []
    for p in parts:
        if '-' in p:
            a, b = p.split('-', 1)
            a = int(a.strip()); b = int(b.strip())
            if a <= b:
                nums.extend(list(range(a, b+1)))
            else:
                nums.extend(list(range(b, a+1)))
        else:
            nums.append(int(p))
    return nums

def _extract_ordered_ids(text: str) -> List[int]:
    """按出现顺序（含重复）提取方括号内的所有数字。"""
    order: List[int] = []
    for m in BRACKET_PATTERN.finditer(text):
        try:
            order.extend(_parse_bracket_content(m.group(1)))
        except Exception:
            pass
    return order

def _renumber_text_with_mapping(text: str, mapping: Dict[int,int]) -> str:
    """
    用 mapping(old->new) 重写文本中的引用；mapping 中不存在的 old 保留原值。
    尽量合并连续区间为 a-b，否则以逗号展开。
    """
    def repl(m: re.Match) -> str:
        content = m.group(1)
        parts = [p.strip() for p in content.split(',') if p.strip()]
        out_parts = []
        for p in parts:
            if '-' in p:
                a,b = p.split('-',1)
                a = int(a.strip()); b = int(b.strip())
                rng = list(range(a, b+1)) if a<=b else list(range(b, a+1))
                mapped = [mapping.get(x, x) for x in rng]
                try:
                    mi = [int(x) for x in mapped]
                    if len(mi) >= 2 and all(mi[i]+1==mi[i+1] for i in range(len(mi)-1)):
                        out_parts.append(f"{mi[0]}-{mi[-1]}")
                    else:
                        out_parts.append(','.join(str(x) for x in mi))
                except Exception:
                    out_parts.append(','.join(str(x) for x in mapped))
            else:
                orig = int(p)
                out_parts.append(str(mapping.get(orig, orig)))
        return '[' + ','.join(out_parts) + ']'
    return BRACKET_PATTERN.sub(repl, text)

# ---------------- 函数1：合并构建 mapping + 实时重写（非 generator） ----------------
def process_paragraphs_and_build_mapping(
    paragraphs: str,
    prev_mapping: Optional[Dict[int,int]] = None,
    start: int = 1
) -> Tuple[str, Dict[int,int]]:
    """
    逐段处理：遇到正文中尚未出现在 mapping 的 old 编号，按段内出现顺序分配新编号；
    随后立即用当前 mapping 重写该段并收集结果。
    返回 (rewritten_paragraphs, final_mapping).
    - prev_mapping: 可传入已有 old->new 映射（用于增量分配）
    - start: 若 prev_mapping 为 None，从该值开始分配 new id
    """
    mapping: Dict[int,int] = dict(prev_mapping) if prev_mapping else {}
    next_id = max(mapping.values()) + 1 if mapping else start

    # 为本段首次出现的旧编号分配 new id（按段内出现顺序）
    order = _extract_ordered_ids(paragraphs)
    for old in order:
        if old not in mapping:
            mapping[old] = next_id
            next_id += 1
    # 用当前 mapping 重写本段
    rewritten = _renumber_text_with_mapping(paragraphs, mapping)
    return rewritten, mapping

# ---------------- 函数2：最终根据 mapping 筛选并重编号参考文献 ----------------
def finalize_bibliography(bib_text: str, final_mapping: Dict[int,int]) -> Tuple[str, Set[int]]:
    """
    根据 final_mapping (old->new)：
      - 筛选出 bib_text 中 old 在 final_mapping.keys() 的条目
      - 将条目的编号替换为 mapping[old]（new）
    返回 (filtered_and_renumbered_bib_text, missing_old_numbers)
    """
    splits = re.split(r'(?m)(?=^\s*\[\s*\d+\s*\])', bib_text.strip())
    if len(splits) == 1:
        splits = re.split(r'(?m)(?=^\s*\d+\.\s*)', bib_text.strip())

    bib_entries: Dict[int,str] = {}
    for part in splits:
        part = part.strip()
        if not part:
            continue
        m = re.match(r'^\s*\[\s*(\d+)\s*\]\s*(.*)', part, flags=re.S)
        if m:
            bib_entries[int(m.group(1))] = part
            continue
        m2 = re.match(r'^\s*(\d+)\.\s*(.*)', part, flags=re.S)
        if m2:
            bib_entries[int(m2.group(1))] = part
            continue

    missing: Set[int] = set()
    new_to_entry: Dict[int,str] = {}
    for old, new in final_mapping.items():
        if old in bib_entries:
            new_to_entry[new] = bib_entries[old]
        else:
            missing.add(old)

    out_lines: List[str] = []
    for new_idx in sorted(new_to_entry.keys()):
        entry = new_to_entry[new_idx]
        s = re.sub(r'^\s*\[\s*\d+\s*\]', f'[{new_idx}]', entry, count=1)
        s = re.sub(r'^\s*\d+\.\s*', f'{new_idx}. ', s, count=1)
        out_lines.append(s)
    final_refs = "\n# 参考文献:\n\n" + '\n\n'.join(out_lines).strip()
    return final_refs, missing

# ---------------- 简单示例 ----------------
if __name__ == "__main__":
    orig_bib = """
[1] A

[2] B

[3] C

[4] D

[5] E

[6] F

[7] G

[8] H
"""
    paragraphs = [
        "段1: 引用了 [4] 和 [2]。",
        "段2: 引用了 [2] 并且范围 [5-6]。",
        "段3: 引用了 [4] 和 [1]。"
    ]
    print("重写段落:")
    mapping = {}
    for p in paragraphs:
        rewritten, mapping = process_paragraphs_and_build_mapping(p,mapping)
        print(rewritten)
    print("最终 mapping:", mapping)

    final_bib, missing = finalize_bibliography(orig_bib, mapping)
    print("\n最终参考文献:\n", final_bib)
    print("missing:", missing)
