"""
基于正则表达式的快速内容检查器
用于替代大模型检查，提高检查速度
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class RegexChecker:
    """基于正则表达式的内容检查器"""
    
    def __init__(self):
        # 禁用元话语/承接词的正则表达式
        self.meta_discourse_patterns = [
            r'(?i)^(本章|本节|本文|下面将|以下将)',
            r'(?i)(上一章|下一章)',
            r'(?i)(如下|如上|综上|总之|由此可见)',
            r'(?i)（本章完）|本章节完',
            r'(?i)(笔者|我们)',
            r'(?i)本章小结',
            r'(?i)在本章中',
            r'(?i)下面将|以下将',
            r'(?i)本研究'
        ]
        
        # 参考文献相关的正则表达式
        self.reference_patterns = [
            r'(?i)^#+\s*(参考文献|References|Bibliography)',  # 独立标题行
            r'(?i)^\*\*(参考文献|References|Bibliography)\*\*',  # 加粗标题
            r'(?i)^(参考文献|References|Bibliography)[:：]',  # 带冒号的标题
            r'doi:\s*10\.\d+',  # DOI格式
            r'https://doi\.org/',  # DOI链接
        ]
        
        # 引用格式检查
        self.citation_patterns = {
            'valid': r'\[\d+\]',  # 正确的单个引用格式
            'invalid_range': r'\[\d+[-–]\d+\]',  # 错误的范围引用
            'invalid_comma': r'\[\d+,\s*\d+\]',  # 错误的逗号分隔引用
            'zero_citation': r'\[0\]',  # 错误的0引用
        }
        
        # 标题格式检查
        self.title_patterns = {
            'chapter': r'^###\s+\d+\s+.+$',  # 章标题格式
            'section1': r'^####\s+\d+\.\d+\s+.+$',  # 一级子节
            'section2': r'^#{5,}\s+\d+\.\d+\.\d+.*\s+.+$',  # 二级及更深层
        }

    def check_content(self, content: str, title: str = "", struct: str = "", 
                     language: str = "中文", block_type: str = "SECTION") -> Dict[str, any]:
        """
        检查内容格式是否合规
        
        Args:
            content: 要检查的内容
            title: 文档标题
            struct: 结构信息
            language: 语言要求
            block_type: 块类型 (ABSTRACT/SECTION)
            
        Returns:
            检查结果字典，包含是否合格和具体问题
        """
        issues = []
        is_valid = True
        
        try:
            if block_type == "ABSTRACT":
                abstract_issues = self._check_abstract_format(content, title)
                issues.extend(abstract_issues)
            else:
                section_issues = self._check_section_format(content, struct)
                issues.extend(section_issues)
            
            # 通用检查
            common_issues = self._check_common_issues(content, block_type)
            issues.extend(common_issues)
            
            # 判断是否有阻断性问题
            blocking_issues = [issue for issue in issues if issue.get('blocking', False)]
            if blocking_issues:
                is_valid = False
                
        except Exception as e:
            logger.error(f"检查过程中出现错误: {e}")
            issues.append({
                'type': 'error',
                'message': f'检查过程中出现错误: {str(e)}',
                'blocking': True
            })
            is_valid = False
        
        result = {
            'valid': is_valid,
            'issues': issues,
            'summary': self._generate_summary(is_valid, issues)
        }
        
        return result

    def _check_abstract_format(self, content: str, title: str) -> List[Dict]:
        """检查摘要格式"""
        issues = []
        lines = content.strip().split('\n')
        
        # 检查四行骨架结构
        if len(lines) < 4:
            issues.append({
                'type': 'structure',
                'message': '摘要缺少必要的四行骨架结构',
                'blocking': True
            })
            return issues
        
        # 检查题目行
        if not lines[0].startswith('# '):
            issues.append({
                'type': 'title',
                'message': '题目行格式错误，应以"# "开头',
                'blocking': True
            })
        
        # 检查摘要行
        abstract_line_found = False
        keywords_line_found = False
        
        for i, line in enumerate(lines):
            if line.strip() == '**摘要**':
                abstract_line_found = True
            elif line.strip().startswith('**关键词'):
                keywords_line_found = True
        
        if not abstract_line_found:
            issues.append({
                'type': 'structure',
                'message': '缺少独立的"**摘要**"行',
                'blocking': True
            })
        
        if not keywords_line_found:
            issues.append({
                'type': 'structure',
                'message': '缺少"**关键词**"行',
                'blocking': True
            })
        
        # 检查摘要中是否有引用（不允许）
        citation_matches = re.findall(self.citation_patterns['valid'], content)
        if citation_matches:
            issues.append({
                'type': 'citation',
                'message': '摘要中不允许出现引用标注',
                'blocking': True
            })
        
        return issues

    def _check_section_format(self, content: str, struct: str) -> List[Dict]:
        """检查章节格式"""
        issues = []
        
        # 解析结构信息
        try:
            if struct:
                struct_data = json.loads(struct) if isinstance(struct, str) else struct
                expected_titles = self._extract_expected_titles(struct_data)
                actual_titles = self._extract_actual_titles(content)
                
                # 检查标题匹配
                title_issues = self._check_title_matching(expected_titles, actual_titles)
                issues.extend(title_issues)
        except Exception as e:
            logger.warning(f"结构解析失败: {e}")
        
        return issues

    def _check_common_issues(self, content: str, block_type: str) -> List[Dict]:
        """检查通用问题"""
        issues = []
        
        # 检查禁用元话语
        meta_issues = self._check_meta_discourse(content)
        issues.extend(meta_issues)
        
        # 检查引用格式
        citation_issues = self._check_citation_format(content, block_type)
        issues.extend(citation_issues)
        
        # 检查参考文献列表
        ref_issues = self._check_reference_list(content)
        issues.extend(ref_issues)
        
        return issues

    def _check_meta_discourse(self, content: str) -> List[Dict]:
        """检查禁用的元话语"""
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            for pattern in self.meta_discourse_patterns:
                if re.search(pattern, line_stripped):
                    # 检查是否在标题或段落首句
                    is_title = line_stripped.startswith('#')
                    is_first_sentence = self._is_paragraph_first_sentence(lines, i-1)
                    
                    blocking = is_title or is_first_sentence
                    
                    issues.append({
                        'type': 'meta_discourse',
                        'message': f'第{i}行包含禁用的元话语: {line_stripped[:50]}...',
                        'line': i,
                        'blocking': blocking
                    })
                    break
        
        return issues

    def _check_citation_format(self, content: str, block_type: str) -> List[Dict]:
        """检查引用格式"""
        issues = []
        
        if block_type == "ABSTRACT":
            # 摘要中不允许引用
            citations = re.findall(self.citation_patterns['valid'], content)
            if citations:
                issues.append({
                    'type': 'citation',
                    'message': '摘要中不允许出现引用',
                    'blocking': True
                })
        else:
            # 检查错误的引用格式
            invalid_ranges = re.findall(self.citation_patterns['invalid_range'], content)
            if invalid_ranges:
                issues.append({
                    'type': 'citation',
                    'message': f'发现错误的范围引用格式: {invalid_ranges}',
                    'blocking': False  # 重要问题但不阻断
                })
            
            invalid_commas = re.findall(self.citation_patterns['invalid_comma'], content)
            if invalid_commas:
                issues.append({
                    'type': 'citation',
                    'message': f'发现错误的逗号分隔引用格式: {invalid_commas}',
                    'blocking': False
                })
            
            zero_citations = re.findall(self.citation_patterns['zero_citation'], content)
            if zero_citations:
                issues.append({
                    'type': 'citation',
                    'message': '发现错误的[0]引用',
                    'blocking': False
                })
        
        return issues

    def _check_reference_list(self, content: str) -> List[Dict]:
        """检查是否包含参考文献列表（阻断性）"""
        issues = []
        
        # 检查参考文献标题
        for pattern in self.reference_patterns[:3]:  # 标题相关的模式
            if re.search(pattern, content, re.MULTILINE):
                issues.append({
                    'type': 'reference_list',
                    'message': '内容中包含参考文献标题，这是不允许的',
                    'blocking': True
                })
                break
        
        # 检查DOI出现次数
        doi_count = len(re.findall(self.reference_patterns[3], content)) + \
                   len(re.findall(self.reference_patterns[4], content))
        
        if doi_count >= 2:
            issues.append({
                'type': 'reference_list',
                'message': f'发现{doi_count}个DOI，疑似包含参考文献条目',
                'blocking': True
            })
        
        return issues

    def _extract_expected_titles(self, struct_data: Dict) -> List[str]:
        """从结构数据中提取期望的标题"""
        titles = []
        
        def extract_recursive(data, level=3):
            if isinstance(data, dict):
                if 'id' in data and 'title' in data:
                    prefix = '#' * level
                    titles.append(f"{prefix} {data['id']} {data['title']}")
                
                if 'subsections' in data:
                    for subsection in data['subsections']:
                        extract_recursive(subsection, level + 1)
            elif isinstance(data, list):
                for item in data:
                    extract_recursive(item, level)
        
        extract_recursive(struct_data)
        return titles

    def _extract_actual_titles(self, content: str) -> List[str]:
        """从内容中提取实际的标题"""
        titles = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                titles.append(line)
        
        return titles

    def _check_title_matching(self, expected: List[str], actual: List[str]) -> List[Dict]:
        """检查标题匹配"""
        issues = []
        
        if len(expected) != len(actual):
            issues.append({
                'type': 'structure',
                'message': f'标题数量不匹配，期望{len(expected)}个，实际{len(actual)}个',
                'blocking': True
            })
        
        for i, (exp, act) in enumerate(zip(expected, actual)):
            if exp.strip() != act.strip():
                issues.append({
                    'type': 'structure',
                    'message': f'第{i+1}个标题不匹配，期望: {exp}，实际: {act}',
                    'blocking': True
                })
        
        return issues

    def _is_paragraph_first_sentence(self, lines: List[str], line_index: int) -> bool:
        """判断是否为段落首句"""
        if line_index == 0:
            return True
        
        # 检查前一行是否为空行或标题
        prev_line = lines[line_index - 1].strip()
        return prev_line == '' or prev_line.startswith('#')

    def _generate_summary(self, is_valid: bool, issues: List[Dict]) -> str:
        """生成检查结果摘要"""
        if is_valid:
            return "合格"
        else:
            blocking_count = len([issue for issue in issues if issue.get('blocking', False)])
            total_count = len(issues)
            return f"不合格，发现{total_count}个问题，其中{blocking_count}个阻断性问题"


def quick_check(content: str, title: str = "", struct: str = "", 
                language: str = "中文", block_type: str = "SECTION") -> str:
    """
    快速检查函数，返回简单的合格/不合格结果
    用于直接替代原有的大模型检查
    """
    checker = RegexChecker()
    result = checker.check_content(content, title, struct, language, block_type)
    
    if result['valid']:
        return "合格"
    else:
        return f"不合格"
        # return f"不合格\n原因：{result['summary']}"