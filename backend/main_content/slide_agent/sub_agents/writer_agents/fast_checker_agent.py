"""
基于正则表达式的快速CheckerAgent
用于替代大模型检查，显著提高检查速度
"""

import time
from typing import Dict, AsyncGenerator
from google.genai import types
from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.callback_context import CallbackContext
from .regex_checker import RegexChecker
from . import index_filter
from . import idx_sort

import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "checker.log"),
    encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
logger.addHandler(file_handler)

class FastCheckerAgent(BaseAgent):
    """基于正则表达式的快速检查Agent"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="WriterCheckerAgent",
            description="基于正则表达式的快速内容格式检查器",
            **kwargs
        )
        # 在初始化后设置regex_checker
        object.__setattr__(self, 'regex_checker', RegexChecker())

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """执行快速检查逻辑"""
        start_time = time.time()
        
        # 获取状态信息
        current_part_index: int = ctx.session.state.get("current_part_index", 0)
        last_draft = ctx.session.state.get("last_draft", "")
        last_struct = ctx.session.state.get("last_struct", "{}")
        language = ctx.session.state.get("language")
        title = ctx.session.state.get("title", "")
        rewrite_retry_count_map: Dict[int, int] = ctx.session.state.get("rewrite_retry_count_map", {})
        idx_mapping = ctx.session.state.get("idx_mapping", {})
        
        
        # 摘要不检查，直接通过
        if current_part_index == 0:
            ctx.session.state["checker_result"] = True
            yield Event(author=self.name, content=types.Content(parts=[types.Part(text=last_draft)]))
            return
        
        try:
            # 确定块类型
            block_type = "ABSTRACT" if current_part_index == 0 else "SECTION"
            
            # 执行正则表达式检查
            check_result = self.regex_checker.check_content(
                content=last_draft,
                title=title,
                struct=last_struct,
                language=language,
                block_type=block_type
            )
            
            # 记录检查耗时
            check_time = time.time() - start_time
            
            # 根据检查结果决定是否通过
            if check_result['valid']:
                # 检查通过
                ctx.session.state["checker_result"] = True
                
                # 处理文本过滤和索引映射
                filter_text, new_mapping = index_filter.process_paragraphs_and_build_mapping(
                    paragraphs=last_draft, 
                    prev_mapping=idx_mapping
                )
                # logger.info(f":第一次检查通过文本，过滤前==========>:\n {filter_text}")
                filter_text = idx_sort.sort_citation_numbers_in_text(filter_text)
                table_text = self._process_table_replacement(filter_text, ctx)
                last_text = self._process_citation_replacement(table_text, ctx)
                # logger.info(f":第一次检查通过文本，过滤后❤️==========>:\n {last_text}")
                ctx.session.state["idx_mapping"] = new_mapping
                yield Event(author=self.name, content=types.Content(parts=[types.Part(text=last_text)]))
                
            else:
                # 检查不通过
                
                retry_count = rewrite_retry_count_map.get(current_part_index, 0)
                if retry_count < 3:
                    ctx.session.state["checker_result"] = False
                    
                    # # 可选：返回详细的检查结果给前端用于调试
                    # error_details = self._format_error_details(check_result)
                    # yield Event(author=self.name, content=types.Content(parts=[types.Part(text=error_details)]))
                    
                else:
                    ctx.session.state["checker_result"] = True
                    
                    filter_text, new_mapping = index_filter.process_paragraphs_and_build_mapping(
                        paragraphs=last_draft, 
                        prev_mapping=idx_mapping
                    )
                    # logger.info(f":第三次检查通过文本，过滤前==========>:\n {filter_text}")
                    filter_text = idx_sort.sort_citation_numbers_in_text(filter_text)
                    table_text = self._process_table_replacement(filter_text, ctx)
                    last_text = self._process_citation_replacement(table_text, ctx)
                    
                    # logger.info(f":第三次检查通过文本，过滤后❤️==========>:\n {last_text}")
                    ctx.session.state["idx_mapping"] = new_mapping
                    yield Event(author=self.name, content=types.Content(parts=[types.Part(text=last_text)]))
                    
        except Exception as e:
            # 出现错误时，默认通过检查以避免阻塞流程
            ctx.session.state["checker_result"] = True
            yield Event(author=self.name, content=types.Content(parts=[types.Part(text=last_draft)]))

    def _process_table_replacement(self, filter_text: str, ctx: InvocationContext) -> str:
        """处理表格替换逻辑的私有方法
        
        只替换表格标题中的"表X"，不替换正文中的表格引用
        
        Args:
            filter_text: 需要处理的文本
            ctx: 调用上下文
            
        Returns:
            处理后的文本
        """
        import re
        
        # 获取当前的表格编号，如果不存在则从1开始
        current_table_number = ctx.session.state.get("table_number", 1)
        logger.info(f"当前表格编号: {current_table_number}")
        
        # 精确匹配表格标题格式的正则表达式
        # 严格匹配格式：表X 标题（X 为字面字符 X）
        # 匹配以下情况：
        # 1. 行首或换行后的"表X "，后面必须跟着标题内容
        # 2. 表X后面必须有1个或2个空格，然后是标题文字
        table_title_pattern = r'(^|\n)(\s*)(表X\s{1,2})(.+)'
        
        matches = list(re.finditer(table_title_pattern, filter_text, re.MULTILINE))
        
        if not matches:
            # 如果没有找到表格标题标记，直接返回原文本
            logger.info("未找到表格标题标记")
            return filter_text
        
        # 记录找到的表格标题数量
        table_count = len(matches)
        logger.info(f"在文本中找到 {table_count} 个表格标题标记")
        
        # 从后往前替换，避免位置偏移问题
        result_text = filter_text
        for i, match in enumerate(reversed(matches)):
            # 计算当前表格的编号（从后往前，所以要反向计算）
            table_num = current_table_number + (table_count - 1 - i)
            
            # 获取匹配的各个部分
            line_start = match.group(1)  # 行首或换行符
            spaces = match.group(2)      # 可能的空格
            table_text = match.group(3)  # "表X " 或 "表X  "（1个或2个空格）
            title_content = match.group(4)  # 标题内容
            
            # 构建替换文本，保持原有的格式和标题内容，统一使用1个空格
            replacement = f"{line_start}{spaces}表{table_num} {title_content}"
            
            # 替换当前匹配的表格标记
            start, end = match.span()
            result_text = result_text[:start] + replacement + result_text[end:]
            
            logger.info(f"替换表格标题标记 '{table_text}{title_content}' -> '表{table_num} {title_content}' (位置: {start}-{end})")
        
        # 更新下一个表格编号到状态中
        ctx.session.state["table_number"] = current_table_number + table_count
        logger.info(f"更新后表格编号: {ctx.session.state['table_number']}")
        
        return result_text

    def _process_citation_replacement(self, text: str, ctx: InvocationContext) -> str:
        """处理引用标注替换逻辑的私有方法
        
        将文本中的规范引用格式 [数字] 转换为脚注引用格式 [^数字]
        
        Args:
            text: 需要处理的文本
            ctx: 调用上下文
            
        Returns:
            处理后的文本
        """
        import re
        
        # 匹配规范的引用格式：[数字]
        pattern = r'\[(\d+)\]'
        
        def replace_citation(match):
            number = match.group(1)
            new_citation = f'[^{number}]'
            logger.info(f"替换引用标注 {match.group()} -> {new_citation}")
            return new_citation
        
        # 使用正则替换所有匹配的引用标注
        result_text = re.sub(pattern, replace_citation, text)
        
        return result_text

    def _format_error_details(self, check_result: Dict) -> str:
        """格式化错误详情用于调试"""
        details = [f"检查结果: {check_result['summary']}"]
        
        if check_result.get('issues'):
            details.append("\n具体问题:")
            for i, issue in enumerate(check_result['issues'], 1):
                issue_type = issue.get('type', 'unknown')
                message = issue.get('message', '未知问题')
                blocking = "阻断性" if issue.get('blocking', False) else "提示性"
                details.append(f"{i}. [{issue_type}] {message} ({blocking})")
        
        return "\n".join(details)


def fast_checker_before_agent_callback(callback_context: CallbackContext) -> None:
    """快速检查器的前置回调"""
    start_time = time.time()
    callback_context.state["fast_check_start_time"] = start_time
    current_part_index = callback_context.state.get("current_part_index", 0)
    logger.info(f"=====>>>8. FastCheckerAgent 开始检查第 {current_part_index} 块")


def fast_checker_after_agent_callback(callback_context: CallbackContext) -> None:
    """快速检查器的后置回调"""
    start_time = callback_context.state.get("fast_check_start_time", time.time())
    cost_time = time.time() - start_time
    total_time = callback_context.state.get("fast_check_total_time", 0.0) + cost_time
    callback_context.state["fast_check_total_time"] = total_time
    current_part_index = callback_context.state.get("current_part_index", 0)
    logger.info(f"=====>>>11. FastCheckerAgent 第{current_part_index}块检查完毕, 耗时: {cost_time:.3f} 秒, 总耗时: {total_time:.3f} 秒")


# 创建快速检查器实例
fast_checker_agent = FastCheckerAgent(
    before_agent_callback=fast_checker_before_agent_callback,
    after_agent_callback=fast_checker_after_agent_callback,
)