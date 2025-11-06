import json
import os
import time
import logging
from typing import Dict, List, Any, AsyncGenerator, Optional, Union
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents import LoopAgent, BaseAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from .tools import SearchImage, DocumentSearch, AbstractSearch
from .utils import stringify_references
from ...config import CONTENT_WRITER_AGENT_CONFIG
from ...create_model import create_model
from . import prompt
from . import index_filter
from . import rotator_logger
from .fast_checker_agent import fast_checker_agent

# 配置：日志目录、前缀、保留天数
LOG_DIR = "logs"
PREFIX = "writer_agent"
KEEP_DAYS = 7  # 保留最近 7 天；改成 0 测试会删除今天之前的所有历史文件
logger = rotator_logger.setup_daily_logger(log_dir=LOG_DIR, prefix=PREFIX, keep_days=KEEP_DAYS)

def my_before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    agent_name = callback_context.agent_name
    history_length = len(llm_request.contents)
    metadata = callback_context.state.get("metadata")
    print(f"调用了{agent_name}模型前的callback, 现在Agent共有{history_length}条历史记录,metadata数据为：{metadata}")
    logger.info(f"=====>>>1. 调用了{agent_name}.my_before_model_callback, 现在Agent共有{history_length}条历史记录,metadata数据为：{metadata}\n历史会话为：{llm_request.contents}")

    return None

def my_after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    # 1. 检查用户输入，注意如果是llm的stream模式，那么response_data的结果是一个token的结果，还有可能是工具的调用
    agent_name = callback_context.agent_name
    response_parts = llm_response.content.parts
    part_texts =[]
    for one_part in response_parts:
        part_text = getattr(one_part, "text", None)
        if part_text is not None:
            part_texts.append(part_text)
    part_text_content = "\n".join(part_texts)
    metadata = callback_context.state.get("metadata")
    callback_context.state["last_draft"] = part_text_content
    print(f"调用了{agent_name}模型后的callback, 这次模型回复{response_parts}条信息,metadata数据为：{metadata},回复内容是: {part_text_content}")
    logger.info(f"=====>>>2. 调用了{agent_name}.my_after_model_callback,metadata数据为：{metadata},回复内容是: {part_text_content}")

    return None

# --- 1. Custom Callback Functions for PPTWriterSubAgent ---
def my_writer_before_agent_callback(callback_context: CallbackContext) -> None:
    """
    在调用LLM之前，从会话状态中获取当前计划，并格式化LLM输入。
    """
    agent_name = callback_context.agent_name
    current_part_index: int = callback_context.state.get("current_part_index", 0)  # Default to 0
    parts_plan_num = callback_context.state.get("parts_plan_num")
    metadata = callback_context.state.get("metadata")
    logger.info(f"=====>>>3. 调用了{agent_name}.my_writer_before_agent_callback, metadata数据为：{metadata}，当前块索引：{current_part_index}，总分块索引：{parts_plan_num}")
    # 返回 None，继续调用 LLM
    return None


def my_after_agent_callback(callback_context: CallbackContext) -> None:
    """
    在LLM生成内容后，将其存储到会话状态中。供下一页ppt生成使用
    """
    agent_name = callback_context.agent_name
    model_last_output_content = callback_context._invocation_context.session.events[-1]
    response_parts = model_last_output_content.content.parts
    part_texts = []
    for one_part in response_parts:
        part_text = getattr(one_part, "text", None)
        if part_text is not None:
            part_texts.append(part_text)
    part_text_content = "\n".join(part_texts)
    metadata = callback_context.state.get("metadata")
    logger.info(f"=====>>>4. 调用了{agent_name}.my_after_agent_callback,metadata数据为：{metadata},回复内容是: {part_text_content}")
    return None

class WriterSubAgent(LlmAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="WriterSubAgent",
            model=create_model(model=CONTENT_WRITER_AGENT_CONFIG["model"], provider=CONTENT_WRITER_AGENT_CONFIG["provider"]),
            description="综述撰写助手，根据要求撰写综述的每一块内容。因为你写的是一块内容，不要在末尾添加参考文献列表。",
            instruction=self._get_dynamic_instruction,
            before_agent_callback=my_writer_before_agent_callback,
            after_agent_callback=my_after_agent_callback,
            before_model_callback=my_before_model_callback,
            after_model_callback=my_after_model_callback,
            tools=[DocumentSearch, AbstractSearch],
            **kwargs
        )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        parts_plan_num: int = ctx.session.state.get("parts_plan_num")
        current_part_index: int = ctx.session.state.get("current_part_index", 0)
        rewrite_retry_count_map = ctx.session.state.get("rewrite_retry_count_map", {})
        # 清空历史记录，防止历史记录进行干扰
        if int(rewrite_retry_count_map.get(current_part_index, 0)) > 0:
            logger.info(f"=====>>>6. 当前正在进行对: 第{current_part_index}个块重新生成")
            del_history = ctx.session.events.pop()
            logger.info(f"=============>>>删除了最后1个内容块：\n{del_history}")
            logger.info(f"=============>>>删除后的历史记录为：\n{ctx.session.events}")
        else:
            logger.info(f"=====>>>6. 当前计划块数{parts_plan_num}, 正在生成第{current_part_index}块内容，清空历史记录")
            ctx.session.events = []
        logger.info(f"=====>>>7. 总的计划块数{parts_plan_num}, 正在生成第{current_part_index}块内容...")
        # 调用父类逻辑（最终结果）
        current_part_index: int = ctx.session.state.get("current_part_index", 0)
        if current_part_index == 0:
            last_struct = ctx.session.state["abstract"]
        else:
            sections = ctx.session.state["sections"]
            last_struct = sections[current_part_index - 1]
        ctx.session.state["last_struct"] = last_struct
        async for event in super()._run_async_impl(ctx):
            print(f"{self.name} 收到事件：{event}")
            logger.info(f"=====>>>5. {self.name} 收到事件：{event}")
            # 不返回结果给前端，等待审核通过后返回
            yield event

    def _get_dynamic_instruction(self, ctx: InvocationContext) -> str:
        current_part_index: int = ctx.state.get("current_part_index", 0)
        
        # 获取language参数
        language = ctx.state.get("language")
        
        if current_part_index == 0:
            current_type = "abstract"
            abstract_outline = ctx.state["abstract"]
            print(f"准备生成第一部分，摘要内容:{abstract_outline}")
            part_prompt = prompt.prompt_mapper[current_type]
            title = ctx.state["title"]
            prompt_instruction = prompt.PREFIX_PROMPT.format(TITLE=title, language=language) + part_prompt.format(ABSTRACT_STRUCT=abstract_outline, TITLE=title,language=language)
        else:
            current_type = "body"
            sections = ctx.state["sections"]
            print(f"总的块数是{len(sections)}，要进行生成的是第{current_part_index-1}块")
            section_outline = sections[current_part_index-1]  # 因为abstract占据了1个索引，所以需要去掉1
            print(f"准备生成正文的第{current_part_index-1}块内容: {section_outline}")
            # 当前生成文章的这块的要求汇入prompt
            # 根据不同的类型，形成不同的prompt
            part_prompt = prompt.prompt_mapper[current_type]
            title = ctx.state["title"]
            existing_text = ctx.state["existing_text"]
            prompt_instruction = prompt.PREFIX_PROMPT.format(TITLE=title, language=language) + part_prompt.format(SECTION_STRUCT=section_outline,language=language)
        print(f"第{current_part_index}块的prompt是：{prompt_instruction}")
        return prompt_instruction


class ControllerAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="ControllerAgent",
            description="根据 Checker 结果决定是否提交并推进到下一块，或要求重写。",
            **kwargs
        )

    async def _run_async_impl(self, ctx: InvocationContext):
        max_retries = 3
        parts_plan_num: int = ctx.session.state.get("parts_plan_num")
        idx: int = ctx.session.state.get("current_part_index", 0)
        checker_result = ctx.session.state.get("checker_result")
        retry_map = ctx.session.state.get("rewrite_retry_count_map", {})
        idx_mapping = ctx.session.state.get("idx_mapping", {})

        if checker_result is True:
            # ✅ 通过：提交并推进
            sections = ctx.session.state.get("existing_sections", [])
            sections.append(ctx.session.state.get("last_draft", ""))
            ctx.session.state["existing_sections"] = sections
            ctx.session.state["existing_text"] = "\n".join(sections)
            ctx.session.state["current_part_index"] = idx + 1
            test_sections1 = ctx.session.state.get("existing_sections", [])
            logger.info(f"====================================>>>当块通过时，检查此时sections的内容：{test_sections1}")
            if idx + 1 == parts_plan_num:
                references = ctx.session.state.get("references", {})
                refs_text = stringify_references(references=references)
                if refs_text:
                    final_bib, missing = index_filter.finalize_bibliography(bib_text=refs_text, final_mapping=idx_mapping)
                    logger.info(f"=====>>>12. 最后一条消息结束，有参考资料，即将发送给请求端：{final_bib}")
                    yield Event(author=self.name, content=types.Content(parts=[types.Part(text=final_bib)]))
                else:
                    logger.info(f"=====>>>12. 注意：最后一块内容已经撰写完成，但是参考引用为空，请检查搜索引擎是否正常。")
                yield Event(author=self.name, actions=EventActions(escalate=True))
            else:
                logger.info(f"=====>>>13. 第 {idx} 块校验通过，进入第 {idx+1} 块。")

        else:
            # ✅ 未通过但还可重试：根据 Checker 已经写好的 retry_map 判断
            count = int(retry_map.get(idx, 0))
            if count >= max_retries:
                warn = f"第 {idx} 块重试超过 {max_retries} 次，将保留最近草稿（可能仍不完全合规）。"
                sections = ctx.session.state.get("existing_sections", [])
                sections.append(ctx.session.state.get("last_draft", ""))
                ctx.session.state["existing_sections"] = sections
                ctx.session.state["existing_text"] = "\n".join(sections)
                ctx.session.state["current_part_index"] = idx + 1
                test_sections = ctx.session.state.get("existing_sections", [])
                logger.info(f"====================================>>>当块达到最大尝试次数时，检查此时sections的内容：{test_sections}")
                if idx + 1 == parts_plan_num:
                    references = ctx.session.state.get("references", {})
                    refs_text = stringify_references(references=references)
                    if refs_text:
                        final_bib, missing = index_filter.finalize_bibliography(bib_text=refs_text, final_mapping=idx_mapping)
                        yield Event(author=self.name, content=types.Content(parts=[types.Part(text=final_bib)]))
                    logger.info(f"=====>>>12. warning: {warn}")
                    yield Event(author=self.name, actions=EventActions(escalate=True))
                else:
                    logger.info(f"=====>>>12. warning: {warn}")
            else:
                retry_map[idx] = count + 1
                ctx.session.state["rewrite_retry_count_map"] = retry_map
                logger.info(f"=====>>>13. 第 {idx} 块未通过，将触发重写（第 {count + 1} 次重试）。")

def my_super_before_agent_callback(callback_context: CallbackContext):
    """
    在Loop Agent调用之前，进行数据处理
    :param callback_context:
    :return:
    """
    title = callback_context.state.get("title","")
    sections = callback_context.state.get("sections","")
    abstract = callback_context.state.get("abstract","")
    logger.info(f"=================>>>提取到的标题为：{title}")
    logger.info(f"=================>>>提取到的章节为：{sections}")
    logger.info(f"=================>>>提取到的摘要为：{abstract}")

    callback_context.state["existing_text"] = ""  #已经生成的内容
    callback_context.state["idx_mapping"] = {}  #新旧索引映射
    # 初始化重试次数记录
    if "rewrite_retry_count_map" not in callback_context.state:
        callback_context.state["rewrite_retry_count_map"] = {}
    if "existing_text" not in callback_context.state:
        callback_context.state["existing_text"] = ""
    if "existing_sections" not in callback_context.state:
        callback_context.state["existing_sections"] = []
    if "last_struct" not in callback_context.state:
        callback_context.state["last_struct"] = ""
    if "current_part_index" not in callback_context.state:
        callback_context.state["current_part_index"] = 0
    if "idx_mapping" not in callback_context.state:
        callback_context.state["idx_mapping"] = {}
    return None

# --- 4. WriterGeneratorLoopAgent ---

writer_generator_loop_agent = LoopAgent(
    name="WriterGeneratorLoopAgent",
    max_iterations=100,  # 设置一个足够大的最大迭代次数，以防万一。主要依赖ConditionAgent停止。
    sub_agents=[
        WriterSubAgent(),   # 生成草稿 -> state["last_draft"]
        fast_checker_agent,     # 校验 -> state["checker_result"] (可选择快速或传统检查器)
        ControllerAgent(),  # 决策：提交/递增/终止 或 触发重写
    ],
    before_agent_callback=my_super_before_agent_callback,
)
