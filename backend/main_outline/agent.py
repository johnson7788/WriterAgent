import os
import random

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools.tool_context import ToolContext
from google.adk.tools import BaseTool
from typing import Dict, List, Any, AsyncGenerator, Optional, Union
from create_model import create_model
from tools import DocumentSearch
from dotenv import load_dotenv
import prompt
load_dotenv()

model = create_model(model=os.environ["LLM_MODEL"], provider=os.environ["MODEL_PROVIDER"])

def before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    # 1. 检查用户输入
    agent_name = callback_context.agent_name
    history_length = len(llm_request.contents)
    metadata = callback_context.state.get("metadata")
    language = metadata.get("language","chinese")
    callback_context.state["language"] = language
    
    print(f"调用了{agent_name}模型前的callback, 现在Agent共有{history_length}条历史记录,metadata数据为：{metadata}")
    print(f"使用的语言参数: {language}")
    #清空contents,不需要上一步的拆分topic的记录, 不能在这里清理，否则，每次调用工具都会清除记忆，白操作了
    # llm_request.contents.clear()
    # 返回 None，继续调用 LLM
    return None
def after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    # 1. 检查用户输入，注意如果是llm的stream模式，那么response_data的结果是一个token的结果，还有可能是工具的调用
    agent_name = callback_context.agent_name
    response_parts = llm_response.content.parts
    part_texts =[]
    for one_part in response_parts:
        part_text = one_part.text
        if part_text is not None:
            part_texts.append(part_text)
    part_text_content = "\n".join(part_texts)
    metadata = callback_context.state.get("metadata")
    print(f"调用了{agent_name}模型后的callback, 这次模型回复{response_parts}条信息,metadata数据为：{metadata},回复内容是: {part_text_content}")
    #清空contents,不需要上一步的拆分topic的记录, 不能在这里清理，否则，每次调用工具都会清除记忆，白操作了
    # llm_request.contents.clear()
    # 返回 None，继续调用 LLM
    return None

def after_tool_callback(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext, tool_response: Dict
) -> Optional[Dict]:

  tool_name = tool.name
  print(f"调用了{tool_name}工具后的callback, tool_response数据为：{tool_response}")
  return None

def get_dynamic_instruction(callback_context: CallbackContext) -> str:
    """
    动态生成指令，根据上下文状态调整指令内容
    """
    metadata = callback_context.state.get("metadata", {})
    language = metadata.get("language", "chinese")
    select_time = metadata.get("select_time", None)
    
    # 基础指令
    base_instruction = prompt.OUTLINE_INSTRUCTION.format(language=language)
    
    print(f"动态生成的指令长度: {len(base_instruction)} 字符")
    return base_instruction

def before_agent_callback(callback_context: CallbackContext) -> None:
    """
    在Agent执行前的回调函数
    """
    agent_name = callback_context.agent_name
    metadata = callback_context.state.get("metadata", {})
    print(f"Agent {agent_name} 开始执行，metadata: {metadata}")
    return None

def after_agent_callback(callback_context: CallbackContext) -> None:
    """
    在Agent执行后的回调函数
    """
    agent_name = callback_context.agent_name
    metadata = callback_context.state.get("metadata", {})
    print(f"Agent {agent_name} 执行完成，metadata: {metadata}")
    return None

root_agent = LlmAgent(
    name="outline_agent",
    model=model,
    description=(
        "generate outline"
    ),
    instruction=get_dynamic_instruction,  # 使用动态指令函数
    before_agent_callback=before_agent_callback,
    after_agent_callback=after_agent_callback,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    after_tool_callback=after_tool_callback,
    tools=[DocumentSearch],
)
