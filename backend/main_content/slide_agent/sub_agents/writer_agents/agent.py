import json
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
from ...config import CONTENT_WRITER_AGENT_CONFIG, CHECKER_AGENT_CONFIG
from ...create_model import create_model
from . import prompt

logger = logging.getLogger(__name__)

def my_before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    # 1. 检查用户输入
    agent_name = callback_context.agent_name
    history_length = len(llm_request.contents)
    metadata = callback_context.state.get("metadata")
    print(f"调用了{agent_name}模型前的callback, 现在Agent共有{history_length}条历史记录,metadata数据为：{metadata}")
    logger.info(f"调用了{agent_name}模型前的callback, 现在Agent共有{history_length}条历史记录,metadata数据为：{metadata}")
    #清空contents,不需要上一步的拆分topic的记录, 不能在这里清理，否则，每次调用工具都会清除记忆，白操作了
    # llm_request.contents.clear()
    # 返回 None，继续调用 LLM
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
    logger.info(f"调用了{agent_name}模型后的callback, 这次模型回复{response_parts}条信息,metadata数据为：{metadata},回复内容是: {part_text_content}")
    #清空contents,不需要上一步的拆分topic的记录, 不能在这里清理，否则，每次调用工具都会清除记忆，白操作了
    # llm_request.contents.clear()
    # 返回 None，继续调用 LLM
    return None

# --- 1. Custom Callback Functions for PPTWriterSubAgent ---
def my_writer_before_agent_callback(callback_context: CallbackContext) -> None:
    """
    在调用LLM之前，从会话状态中获取当前计划，并格式化LLM输入。
    """
    current_part_index: int = callback_context.state.get("current_part_index", 0)  # Default to 0
    parts_plan_num = callback_context.state.get("parts_plan_num")
    # 返回 None，继续调用 LLM
    return None


def my_after_agent_callback(callback_context: CallbackContext) -> None:
    """
    在LLM生成内容后，将其存储到会话状态中。供下一页ppt生成使用
    """
    model_last_output_content = callback_context._invocation_context.session.events[-1]
    response_parts = model_last_output_content.content.parts
    part_texts = []
    for one_part in response_parts:
        part_text = getattr(one_part, "text", None)
        if part_text is not None:
            part_texts.append(part_text)
    part_text_content = "\n".join(part_texts)
    # 获取或初始化存储所有生成内容的列表
    existing_sections: List[str] = callback_context.state.get("existing_sections", [])
    existing_sections.append(part_text_content)

    # 更新会话状态
    callback_context.state["existing_sections"] = existing_sections
    callback_context.state["existing_text"] = "\n".join(existing_sections)
    print(f"--- Stored content for {callback_context.state.get('current_part_index', 0) + 1} ---")

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
            print(f"当前正在进行对: {current_part_index}个块重新生成，不需要清空历史记录")
            rewrite_hint = ctx.session.state.get("rewrite_hint")
            feedback_text = (
                "【重写提示】以下为上轮审查反馈（rewrite_hint）。\n"
                "请严格依据此反馈对本块进行**完整重写**：修正指出的问题，避免复用不合格表述；必要时可调整结构与例证。\n\n"
                f"{rewrite_hint}"
            )
            ctx.session.events.append(
                Event(
                    author="CheckerFeedback",
                    content=types.Content(parts=[types.Part(text=feedback_text)])
                )
            )
            print(f"为第{current_part_index}块追加重写反馈事件，rewrite_retry_count_map={rewrite_retry_count_map}")
        else:
            print(f"当前计划块数{parts_plan_num}, 正在生成第{current_part_index}块内容，清空历史记录")
            ctx.session.events = []
        print(f"总的计划块数{parts_plan_num}, 正在生成第{current_part_index}块内容...")
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
            logger.info(f"{self.name} 收到事件：{event}")
            # 不返回结果给前端，等待审核通过后返回
            yield event

    def _get_dynamic_instruction(self, ctx: InvocationContext) -> str:
        current_part_index: int = ctx.state.get("current_part_index", 0)
        current_part_index: int = ctx.state.get("current_part_index", 0)
        rewrite_retry_count_map = ctx.state.get("rewrite_retry_count_map", {})
        retry_num = int(rewrite_retry_count_map.get(current_part_index, 0))
        if retry_num > 0:
            print(f"当前正在进行对: {current_part_index}个块重新生成，生成后返回给前端")
            prompt_instruction = prompt.FIX_ERROR_PROMPT
        elif current_part_index == 0:
            current_type = "abstract"
            abstract_outline = ctx.state["abstract"]
            print(f"准备生成第一部分，摘要内容:{abstract_outline}")
            part_prompt = prompt.prompt_mapper[current_type]
            title = ctx.state["title"]
            prompt_instruction = prompt.PREFIX_PROMPT.format(TITLE=title) + part_prompt.format(ABSTRACT_STRUCT=abstract_outline, TITLE=title)
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
            prompt_instruction = prompt.PREFIX_PROMPT.format(TITLE=title) + part_prompt.format(SECTION_STRUCT=section_outline)
        print(f"第{current_part_index}块的prompt是：{prompt_instruction}")
        return prompt_instruction


class CheckerAgent(LlmAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        current_part_index: int = ctx.session.state.get("current_part_index", 0)
        last_draft = ctx.session.state.get("last_draft")
        last_struct = ctx.session.state.get("last_struct")
        rewrite_retry_count_map: Dict[int, int] = ctx.session.state.get("rewrite_retry_count_map", {})
        if current_part_index == 0:
            print(f"不检查摘要，直接返回给前端")
            yield Event(author=self.name, content=types.Content(parts=[types.Part(text=last_draft)]))
            ctx.session.state["current_part_index"] += 1
            return
        async for event in super()._run_async_impl(ctx):
            print(f"{self.name} 检查结果事件：{event}")
            result = event.content.parts[0].text.strip()

            if "不合格" in result:
                retry_count = rewrite_retry_count_map.get(current_part_index, 0)
                if retry_count < 3:
                    ctx.session.state["rewrite_reason"] = result
                    print(f"[CheckerAgent] 第 {retry_count + 1} 次尝试重写 slide {current_part_index + 1}")
                    ctx.session.state["current_part_index"] = current_part_index
                    rewrite_retry_count_map[current_part_index] = retry_count + 1
                    ctx.session.state["rewrite_retry_count_map"] = rewrite_retry_count_map
                    ctx.session.state["checker_result"] = False
                    ctx.session.state["rewrite_hint"] = result
                else:
                    print(f"[CheckerAgent] 第 {retry_count} 次重写失败，已达最大次数，使用最后一次的draft的数据")
                    ctx.session.state["rewrite_hint"] = result
                    ctx.session.state["checker_result"] = True
                    yield Event(author=self.name, content=types.Content(parts=[types.Part(text=last_draft)]))
            else:
                # 审核通过了，那么返回WriterSubAgent的回答给前端
                ctx.session.state["checker_result"] = True
                yield Event(author=self.name, content=types.Content(parts=[types.Part(text=last_draft)]))

def checker_before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest):
    start_time = time.time()
    callback_context.state["check_start_time"] = start_time
    return None
def checker_after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse):
    start_time = callback_context.state.get("check_start_time")
    cost_time = time.time() - start_time
    total_time = callback_context.state.get("check_total_time", 0.0) + cost_time
    callback_context.state["check_total_time"] = total_time
    current_part_index = callback_context.state.get("current_part_index", 0)
    agent_name = callback_context.agent_name
    logger.warning(f"调用了{agent_name}模型后的callback, 第{current_part_index + 1}块检查完毕, 耗时: {cost_time} 秒, 总耗时: {total_time} 秒")
    return None

writer_checker_agent = CheckerAgent(
    model=create_model(model=CHECKER_AGENT_CONFIG["model"], provider=CHECKER_AGENT_CONFIG["provider"]),
    name="WriterCheckerAgent",
    description="检查撰写综述章节的内容是否合格",
    instruction=prompt.CHECKER_AGENT_PROMPT,
    before_model_callback=checker_before_model_callback,
    after_model_callback=checker_after_model_callback,
)


class ControllerAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="ControllerAgent",
            description="根据 Checker 结果决定是否提交并推进到下一块，或要求重写。",
            **kwargs
        )

    async def _run_async_impl(self, ctx: InvocationContext):
        max_retries = 1
        parts_plan_num: int = ctx.session.state.get("parts_plan_num")
        idx: int = ctx.session.state.get("current_part_index", 0)
        checker_result = ctx.session.state.get("checker_result")
        retry_map = ctx.session.state.get("rewrite_retry_count_map", {})

        if checker_result is True:
            # ✅ 通过：提交并推进
            sections = ctx.session.state.get("existing_sections", [])
            sections.append(ctx.session.state.get("last_draft", ""))
            ctx.session.state["existing_sections"] = sections
            ctx.session.state["existing_text"] = "\n".join(sections)
            ctx.session.state.pop("rewrite_hint", None)
            ctx.session.state["current_part_index"] = idx + 1

            if idx + 1 == parts_plan_num:
                references = ctx.session.state.get("references", {})
                refs_text = stringify_references(references=references)
                if refs_text:
                    print(f"最后一条消息结束，有参考资料，即将发送给请求端：{refs_text}")
                    yield Event(author=self.name, content=types.Content(parts=[types.Part(text=refs_text)]))
                else:
                    print(f"注意：最后一块内容已经撰写完成，但是参考引用为空，请检查搜索引擎是否正常。")
                yield Event(author=self.name, actions=EventActions(escalate=True))
            else:
                print(f"第 {idx} 块校验通过，进入第 {idx+1} 块。")

        else:
            # ✅ 未通过但还可重试：根据 Checker 已经写好的 retry_map 判断
            count = int(retry_map.get(idx, 0))
            if count > max_retries:
                warn = f"第 {idx} 块重试超过 {max_retries} 次，将保留最近草稿（可能仍不完全合规）。"
                last_draft = ctx.session.state.get("last_draft", "")
                sections = ctx.session.state.get("existing_sections", [])
                sections.append(last_draft)
                ctx.session.state["existing_sections"] = sections
                ctx.session.state["existing_text"] = "\n".join(sections)
                ctx.session.state["current_part_index"] = idx + 1
                ctx.session.state["checker_result"] = True
                ctx.session.state.pop("rewrite_hint", None)

                if last_draft:
                    # 在放弃继续重写时依然把最新草稿返回给上游，避免章节缺失
                    yield Event(
                        author="WriterCheckerAgent",
                        content=types.Content(parts=[types.Part(text=last_draft)])
                    )

                if idx + 1 == parts_plan_num:
                    references = ctx.session.state.get("references", {})
                    refs_text = stringify_references(references=references)
                    if refs_text:
                        yield Event(author=self.name, content=types.Content(parts=[types.Part(text=refs_text)]))
                    print(f"warning: {warn}")
                    yield Event(author=self.name, actions=EventActions(escalate=True))
                else:
                    print(f"warning: {warn}")
            else:
                print(f"第 {idx} 块未通过，将触发重写（第 {count} 次重试）。")

def my_super_before_agent_callback(callback_context: CallbackContext):
    """
    在Loop Agent调用之前，进行数据处理
    :param callback_context:
    :return:
    """
    callback_context.state["existing_text"] = ""  #已经生成的内容
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
    return None

# --- 4. WriterGeneratorLoopAgent ---
writer_generator_loop_agent = LoopAgent(
    name="WriterGeneratorLoopAgent",
    max_iterations=100,  # 设置一个足够大的最大迭代次数，以防万一。主要依赖ConditionAgent停止。
    sub_agents=[
        WriterSubAgent(),   # 生成草稿 -> state["last_draft"]
        writer_checker_agent,     # 校验 -> state["checker_result"]
        ControllerAgent(),  # 决策：提交/递增/终止 或 触发重写
    ],
    before_agent_callback=my_super_before_agent_callback,
)
