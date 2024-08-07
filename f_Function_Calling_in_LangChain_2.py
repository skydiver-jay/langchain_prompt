"""
"Function Calling in LangChain"章节的样例代码
注意：
    ** tool_calls 是 function_call 的进阶版，由于 openai 已将 function_call 等参数（例如 functions）标记为“已废弃”
    ** 由于样例代码后端使用的LLM为MoonShot而非OpanAI官方的model，在LangChain中使用MoonShot & ToolCall无直接的文档参考
        ## 以下代码为综合参考LangChain官方文档以及MoonShot官方文档
            ### MoonShot：https://platform.moonshot.cn/docs/guide/use-kimi-api-to-complete-tool-calls#%E6%89%A7%E8%A1%8C%E5%B7%A5%E5%85%B7
            ### LangChain：https://python.langchain.com/v0.2/docs/how_to/function_calling/
    ** 由于书中的例子不太直观，以下代码使用LangChain官方文档中的加法和乘法为例子
"""
import os
from time import sleep

from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional
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

# Convert Pydantic objects to the appropriate schema:
# 这一步很关键，如果不进行转换，按照LangChain官方文档直接bind_tools()，会不成功
# 更新@20240807，后续观察表明这里参考书中示例代码进行转换是多余的，参考官方文档直接bind_tool()是OK的：
#   1. LangChain提供的bind_tools()方法实现用第一行代码就是 `formatted_tools = [convert_to_openai_tool(tool) for tool in tools]`
#   2. 使用MoonShot Tool Call特性异常不成功最终证明不是该原因导致，而是max token配置设置问题
tools_original = tools
tools = [convert_to_openai_tool(p) for p in tools]  # 就是这一步是多余的
# print(tools)

llm = ChatOpenAI(
    base_url="https://api.moonshot.cn/v1",
    api_key=os.environ["MOONSHOT_API_KEY"],
    model="moonshot-v1-32k",
    max_tokens=20000
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


def test_case_01():
    """测试bind_tools()传入原始数据类型，message为简单query str的场景"""

    """按照MoonShot客服说的在实例化model时设置max token值，8k的模型即使设置为最大值8192也会异常(异常返回详见文件末尾注释)，测试发现32k模型设置20000返回正常"""

    """by 客服：
        您好，每个模型都有最大的长度限制，8k模型的最大长度是8192token，然后32k模型的最大长度是32768token，128k模型的最大长度是131072token呢~
        您的maxtokens设置不要超过这个模型对应长度就好了呢~
    """

    llm_with_tools_original = llm.bind_tools(tools=tools_original)
    print("Tool struct before bind: ", tools_original)
    ai_msg_original = llm_with_tools_original.invoke(query)
    if len(ai_msg_original.tool_calls) != 0:
        # success_count = success_count + 1
        print(f"success, ", ai_msg_original)
    return llm_with_tools_original, ai_msg_original


def test_case_02():
    """测试bind_tools()传入原始数据类型，message为messages对象的场景"""

    """按照MoonShot客服说的在实例化model时设置max token值，8k的模型即使设置为最大值8192也会异常(异常返回详见文件末尾注释)，测试发现32k模型设置20000返回正常"""

    llm_with_tools_original = llm.bind_tools(tools=tools_original)
    print("Tool struct before bind: ", tools_original)
    ai_msg_original = llm_with_tools_original.invoke(messages)
    if len(ai_msg_original.tool_calls) != 0:
        # success_count = success_count + 1
        print(f"success, ", ai_msg_original)
    return llm_with_tools_original, ai_msg_original


def test_case_03():
    """连接test_caes_02()完成tool call，并输出最终答案"""
    llm_with_tools, ai_msg = test_case_02()
    # append ai_msg处有提示类型Warning，但此处代码为参考LangChain官方示例代码:https://python.langchain.com/v0.2/docs/how_to/function_calling 中‘Passing tool outputs to model’部分
    messages.append(ai_msg)
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
    result = llm_with_tools.invoke(messages)

    print(result)
    print(result.content)
    return result


if __name__ == "__main__":
    test_case_03()
    exit(1)

"""
openai.BadRequestError: Error code: 400 - {'error': {'message': 'Invalid request: 
Your request exceeded model token limit: 8192', 'type': 'invalid_request_error'}}
"""
