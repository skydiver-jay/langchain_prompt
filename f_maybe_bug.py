"""
这部分代码是在LangChain中使用MoonShot Tool Call特性遇到的异常记录。
    经过咨询客服，异常原因为：token数量超过了所调用版本的LLM最大token数限制（不敢相信，简单的算术运算问题会触发该上限）
    经过实际测试：MoonShot 8k的模型即使设置为最大值8192也会异常，32k模型设置20000返回正常，"贵还是贵的道理的"，正常代码见f_Function_Calling_in_LangChain_2.py
"""
import os
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage


@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]
tools = [convert_to_openai_tool(p) for p in tools]

llm = ChatOpenAI(
    base_url="https://api.moonshot.cn/v1",
    api_key=os.environ["MOONSHOT_API_KEY"],
    model="moonshot-v1-8k",
)

query = "What is 3 * 12? Also, what is 11 + 49?"

messages = [
    SystemMessage(
        content="你是 Kimi，由 Moonshot AI提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。"
    ),
    HumanMessage(
        content=query
    ),
]

llm_with_tools = llm.bind_tools(tools=tools)

# 这样使用返回正常，finish reason为tool call
# 此用法也会偶尔出现和下面一种用法同样的异常，但不像下面一种，从来没成功过
ai_msg = llm_with_tools.invoke(query)
print(ai_msg)
'''
content='' additional_kwargs={'tool_calls': [{'id': 'multiply:0', 'function': {'arguments': '{\n"a": 3,\n"b": 12\n}', 'name': 'multiply'}, 'type': 'function', 'index': 0}, {'id': 'add:1', 'function': {'arguments': '{\n"a": 11,\n"b": 49\n}', 'name': 'add'}, 'type': 'function', 'index': 1}]} response_metadata={'token_usage': {'completion_tokens': 35, 'prompt_tokens': 96, 'total_tokens': 131}, 'model_name': 'moonshot-v1-8k', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None} id='run-d84f967c-842e-4b20-bd85-f1c3870e09af-0' tool_calls=[{'name': 'multiply', 'args': {'a': 3, 'b': 12}, 'id': 'multiply:0', 'type': 'tool_call'}, {'name': 'add', 'args': {'a': 11, 'b': 49}, 'id': 'add:1', 'type': 'tool_call'}] usage_metadata={'input_tokens': 96, 'output_tokens': 35, 'total_tokens': 131}
'''


# 这样使用返回异常，finish reason为length
ai_msg = llm_with_tools.invoke(messages)
print(ai_msg)
'''
content='' response_metadata={'token_usage': {'completion_tokens': 0, 'prompt_tokens': 131, 'total_tokens': 131}, 'model_name': 'moonshot-v1-8k', 'system_fingerprint': None, 'finish_reason': 'length', 'logprobs': None} id='run-0d08f81c-65c3-42e8-8cd2-1fc3eba35b9e-0' usage_metadata={'input_tokens': 131, 'output_tokens': 0, 'total_tokens': 131}
'''


# 这样使用（不bind tool）也正常
ai_msg = llm.invoke(messages)
print(ai_msg)
'''
content='The multiplication of 3 and 12 equals 36. And the addition of 11 and 49 equals 60.' response_metadata={'token_usage': {'completion_tokens': 27, 'prompt_tokens': 61, 'total_tokens': 88}, 'model_name': 'moonshot-v1-8k', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-1fad33a5-01a5-4a3a-a757-1ad4414f108d-0' usage_metadata={'input_tokens': 61, 'output_tokens': 27, 'total_tokens': 88}
'''
