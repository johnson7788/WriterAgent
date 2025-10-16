import re
from typing import List

BRACKET_RE = re.compile(r'\[\s*([0-9\-,\s]+?)\s*\]')  # 匹配单个 [ ... ]（不含尾随空格）
GROUP_RE = re.compile(r'(?:\[\s*[0-9\-,\s]+?\s*\])+')  # 匹配连续的一个或多个方括号组（如 [1][2] 或 [1,2]）

def _parse_token(tok: str):
    """解析一个单项 token（可能是 '12' 或 '3-5'），返回 (orig_str, start_int)"""
    tok = tok.strip()
    m = re.match(r'^(\d+)\s*-\s*(\d+)$', tok)
    if m:
        a = int(m.group(1)); b = int(m.group(2))
        start = min(a, b)
        # 标准化范围为 "a-b"（小->大）
        return (f"{start}-{max(a,b)}", start)
    m2 = re.match(r'^(\d+)$', tok)
    if m2:
        n = int(m2.group(1))
        return (str(n), n)
    # 非数字 token —— 返回原样并用大数作为排序关键字（放到末尾）
    return (tok, 10**18)

def sort_citation_numbers_in_text(text: str) -> str:
    """
    将文本中每个连续的方括号引用组内部的编号按从小到大排序。
    - 对于 [54,45] -> [45,54]
    - 对于 [54][45] -> [45][54]
    - 对于 [54][45,12] -> [12][45,54]（保留每个 bracket 的项数分布）
    - 已经按升序则保持不变
    """
    def process_group(group_text: str) -> str:
        # 找到该组里所有单个 bracket（不含间隔）
        matches = list(BRACKET_RE.finditer(group_text))
        if not matches:
            return group_text

        # 提取每个 bracket 的 tokens（字符串形式）和 counts
        per_bracket_tokens: List[List[str]] = []
        separators: List[str] = []
        for i, m in enumerate(matches):
            content = m.group(1)
            # 切分逗号为单项
            items = [t.strip() for t in content.split(',') if t.strip() != ""]
            per_bracket_tokens.append(items)
        # 记录原始 separators（bracket 之间的原始间隔，如空格）
        # 我们计算每个匹配项的结尾与下一个匹配项的开头之间相对于group_text的片段
        for i in range(len(matches)):
            end_pos = matches[i].end()
            next_start = matches[i+1].start() if i+1 < len(matches) else len(group_text)
            separators.append(group_text[end_pos:next_start])  # 可能是空字符串或空格，保留以便复原

        # flatten tokens and parse for sorting
        flat = []
        for br_idx, items in enumerate(per_bracket_tokens):
            for item in items:
                orig_str, key = _parse_token(item)
                flat.append((orig_str, key))

        if not flat:
            return group_text  # nothing to do

        # 检查是否已按当前扁平化顺序中的键进行排序
        keys = [k for (_s, k) in flat]
        if keys == sorted(keys):
            return group_text  # 已经在上升, 不变

        # 按关键字排序（稳定排序）
        flat_sorted = sorted(flat, key=lambda x: x[1])

        # 将排序后的项目重新分配到括号中，同时保留每个括号的数量
        counts = [len(items) for items in per_bracket_tokens]
        idx = 0
        new_brackets = []
        for c in counts:
            if c == 0:
                new_brackets.append("[]")
            else:
                items_for_br = [flat_sorted[idx + j][0] for j in range(c)]
                idx += c
                new_brackets.append("[" + ",".join(items_for_br) + "]")

        # 使用已记录的分隔符重新连接（注意分隔符长度等于括号数量）
        rebuilt = ""
        for i, br in enumerate(new_brackets):
            rebuilt += br
            # 添加位于该括号后的分隔符（对于最后一个括号，分隔符是组内的尾部）
            rebuilt += separators[i] if i < len(separators) else ""
        return rebuilt

    # 替换 group（逐个匹配连续 bracket 组）
    out = []
    last_end = 0
    for g in GROUP_RE.finditer(text):
        start, end = g.start(), g.end()
        out.append(text[last_end:start])
        group_text = text[start:end]
        out.append(process_group(group_text))
        last_end = end
    out.append(text[last_end:])
    original_text="".join(out)
    # 分步过滤，先处理连续引用标记和引用列表
    filtered_text = re.sub(
        r'(?:\[\d+\])+\s*引用自：.*?(?=\n\n|\n[^-\s]|$)',  # 过滤连续引用标记和后续引用列表，直到遇到非引用内容
        '', 
        original_text, 
        flags=re.DOTALL | re.MULTILINE)
    
    # 再处理其他引用格式
    filtered_text = re.sub(
        r'(?:---\s*)?引用：(?:\[\d+\])+\s*|'  # 过滤带可选---前缀的引用标记
        r'(?:\n\s*|\s+|(?<=[.。!!]))\[\d+\].*?(?=\n|$)|'  # 过滤行首、空格分隔或句号后的数字引用及内容
        r'\n\s*-\s*\d{4}《.*?》.*?(?=\n|$)|'  # 过滤单独的引用条目行
        r'引用顺序说明：.*?(?=\n\n|\n[^\s\[\-]|$)',  # 过滤"引用顺序说明："及其后的所有内容，直到遇到非引用内容
        '', 
        filtered_text, 
        flags=re.DOTALL | re.MULTILINE).strip() 
    return filtered_text

def replace_table_x(text: str, num: int) -> str:
    """
    将文本中的"表X"和"表 X"替换为"表num"
    
    参数:
        text: 需要处理的文本字符串
        num: 要替换x的整数
        
    返回:
        替换后的文本字符串
    """
    # 使用正则表达式同时匹配"表X"和"表 X"
    # \s? 表示匹配0个或1个空格
    return re.sub(r'表\s?X', f'表{num}', text)
# ---------------- 测试示例 ----------------
if __name__ == "__main__":
    examples = [
        """3.3 风险分层对治疗决策的作用
风险分层工具在个体化治疗策略制定中具有关键作用。以免疫表型及分子遗传学为基础，CLL患者可分为低、中、高风险组，不同风险分组与起始治疗、药物选择及随访频率直接相关[10][11]。国际预后指标（CLL-IPI）整合年龄、临床分期、β2-微球蛋白、IGHV状态及17p/TP53异常，有助于精准界定治疗时机及药物方案[10]。高风险亚组（如TP53突变、IGHV未突变）优先考虑靶向药物如BCL-2抑制剂（如Venetoclax）及BTK抑制剂，而非传统化学免疫疗法[11][12]。部分低风险、无症状患者可继续观察等待（watch and wait），避免过度治疗[11]。新型风险分层模型结合流式细胞免疫分型与分子标志物，在预后评估和动态调整治疗方案中的作用日益凸显[10]。

[5] Review: Revolutionizing chronic lymphocytic leukemia diagnosis: A deep dive into the diverse applications of machine learning.
[2] Case Report: Diagnosis and Management of a Patient With Chronic Lymphocytic Leukemia and a Concurrent Plasmacytoma.
""",
     """
     本节内容侧重于不同亚组和特殊类型乳腺癌患者应用小分子靶向药物时的疗效、安全性和临床管理要点，相关建议需要结合个体临床特征及现有分层证据进行综合判断[8][9][11][12]。
    ---
    引用：[8][9][11][12]

     """,
     """
    研究数据显示，在乳腺癌骨转移病灶组织中，SOST表达水平显著升高，且与骨溶解相关因子协同促进微环境有利于肿瘤生长[14][16]。小分子SOST抑制剂通过阻断该蛋白的活性可明显降低肿瘤在骨组织的生长速率、减少骨破坏，并改善骨转移相关的疼痛与生活质量指标[14]。新型小分子SOST抑制剂配合其他骨保护药物（如双膦酸盐、地诺单抗）显示出协同保护骨结构与抑制肿瘤转移的前景[14][15]。
    引用顺序说明：
    [5][8][9][11][12]：基于“乳腺癌 PI3K mTOR 靶向药物”检索结果
    [1][2][3][4][10]：基于“乳腺癌 程序性细胞死亡 靶点”检索结果
    [6][7][13]：基于“乳腺癌 多靶点 小分子”检索结果
    [14][15][16][17]：基于“乳腺癌 SOST 骨转移”检索结果

     """,
     """
    现有证据基于动物实验与小型临床研究，未来亟需多中心、大样本人群中的随机对照试验，明确不同剂型与递送方式对视觉结局、生物标志物改善及患者依从性的真实获益。[1][2][3][4] 引用自：
[1] 2024《The Role of Natural Products in Diabetic Retinopathy》
[2] 2023《Discussing pathologic mechanisms of Diabetic retinopathy & therapeutic potentials of curcumin and β-glucogallin in the management of Diabetic retinopathy》
     """,
     """
     耐药结核分枝杆菌还表现出遗传多样性高度丰富，不同谱系的耐药相关等位基因传播能力和适应性存在差异，影响其群体进化和全球流行趋势[6][12]。[7] Unraveling the mechanisms of intrinsic drug resistance in Mycobacterium tuberculosis（Front Cell Infect Microbiol, 2022）
     """,
     """
     在病例-对照分析中，TB-LAMP在疑似肺结核患者中的诊断准确率优于部分传统显微镜检查，为结核病早期诊断提供了可行技术路径，尤其适合一线防控和流行病学调查场景[26]。[1][2][3][7][17][18][19][20][21][22][23][24][25][26]
     """,
     """
     FoxO1通路异常所带来的脂解障碍及能量代谢失衡[14][16]。[14] Zhao N, Tan H, Wang L, et al. Palmitate induces fat accumulation via repressing FoxO1-mediated ATGL-dependent lipolysis in HepG2 hepatocytes. PloS one. 2021.
     """,
     """
     表X 小儿结核诊断的主流技术及特色

| 方法              | 特点                     | 敏感性 | 特异性 | 优势                   | 局限                |
|-------------------|--------------------------|--------|--------|------------------------|---------------------|
| 痰涂片/培养        | 核心微生物学标准         | 低     | 高     | 诊断标准，抗药性可判别 | 样本采集难，菌量低  |
| TST/IGRA          | 免疫应答检测             | 中     | 中     | 操作简便               | 活动与潜伏难区分    |
| 血浆/血清生物标志物| 多因子分子水平定量       | 中高   | 中高   | 采样便利，敏感性提升   | 标准化不足，需验证  |
| 支气管镜取材       | 侵入性取材               | 高     | 高     | 针对顽固或复杂病例      | 设备要求高，适应症窄 |
     """,
     """
     环介导等温扩增（Loop-Mediated Isothermal Amplification, LAMP）作为等温核酸扩增技术，具有操作简便、无需昂贵仪器的特点，适合基层和资源有限环境的快速结核病现场筛查[22]。多项研究显示，TB-LAMP法对成人及儿童肺结核具有较高的灵敏度和特异性，在口腔拭子、痰液等多样本型中实现有效检测[23][24][25]。通过靶向IS6110等结核分枝杆菌高保守基因区域，LAMP方法结合金纳米探针等新型检测手段，实现肉眼可见结果判读，简化操作及提升现场可及性[22][25]。在病例-对照分析中，TB-LAMP在疑似肺结核患者中的诊断准确率优于部分传统显微镜检查，为结核病早期诊断提供了可行技术路径，尤其适合一线防控和流行病学调查场景[26]。[1][2][3][7][17][18][19][20][21][22][23][24][25][26]
     """,
     """
     表X 常见结核分枝杆菌及非结核分枝杆菌耐药机制与分子靶点对比

| 类型              | 主要药物            | 常见耐药基因/靶点                | 代表性突变位点或机制            | 备注                       |
|------------------|--------------------|----------------------------------|-------------------------------|----------------------------|
| 结核分枝杆菌      | 利福平              | rpoB                             | S531L, H526Y, D516V           | 影响RNA聚合酶β亚基         |
| 结核分枝杆菌      | 异烟肼              | katG, inhA启动子                | katG S315T, inhA -15C→T       | 涉及药物活化和靶酶表达调控 |
| 结核分枝杆菌      | 喹诺酮类            | gyrA, gyrB                       | A90V, D94G                    | DNA旋转酶亚基              |
| 结核分枝杆菌      | 吡嗪酰胺            | pncA                             | 各种点突变                    | 酶失活型突变种类繁多       |
| 结核分枝杆菌      | 贝达喹啉            | Rv0678, atpE                     | Rv0678等                      | 外排泵调控、靶点突变       |
| 非结核分枝杆菌    | 大环内酯类          | erm(41), 外排泵/修饰酶基因        | erm(41)表达等                  | M. abscessus优势明显         |
| 非结核分枝杆菌    | 多种抗生素          | β-内酰胺酶、质粒相关各类基因      | 结构酶与外排泵                 | 固有耐药及获得性耐药         |

     """
    ]
    for ex in examples:
        print("原文:", ex)
        print("处理:", sort_citation_numbers_in_text(ex))
        print("---")
