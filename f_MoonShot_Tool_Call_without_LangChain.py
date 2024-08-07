import json
import os
from openai import OpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.tools import tool

client = OpenAI(
    api_key=os.environ["MOONSHOT_API_KEY"],  # 在这里将 MOONSHOT_API_KEY 替换为你从 Kimi 开放平台申请的 API Key
    base_url="https://api.moonshot.cn/v1",
)


@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]

tool_map = {
    "add": add,
    "multiply": multiply,
}

# Convert Pydantic objects to the appropriate schema:
tools = [convert_to_openai_tool(p) for p in tools]
print(tools)

messages = [
    {"role": "system",
     "content": "你是 Kimi，由 Moonshot AI "
                "提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI "
                "为专有名词，不可翻译成其他语言。"},
    {"role": "user", "content": "What is 3 * 12? Also, what is 11 + 49?"}  # 在提问中要求 Kimi 大模型联网搜索
]


def demo_register_tools():
    completion = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=messages,
        temperature=0.3,
        tools=tools,  # <-- 我们通过 tools 参数，将定义好的 tools 提交给 Kimi 大模型
    )

    print(completion.choices[0].model_dump_json(indent=4))


def demo_run_tools():
    finish_reason = None
    # 我们的基本流程是，带着用户的问题和 tools 向 Kimi 大模型提问，如果 Kimi 大模型返回了 finish_reason: tool_calls，则我们执行对应的 tool_calls，
    # 将执行结果以 role=tool 的 message 的形式重新提交给 Kimi 大模型，Kimi 大模型根据 tool_calls 结果进行下一步内容的生成：
    #
    #   1. 如果 Kimi 大模型认为当前的工具调用结果已经可以回答用户问题，则返回 finish_reason: stop，我们会跳出循环，打印出 message.content；
    #   2. 如果 Kimi 大模型认为当前的工具调用结果无法回答用户问题，需要再次调用工具，我们会继续在循环中执行接下来的 tool_calls，直到 finish_reason 不再是 tool_calls；
    #
    # 在这个过程中，只有当 finish_reason 为 stop 时，我们才会将结果返回给用户。
    # 详细请见：https://platform.moonshot.cn/docs/guide/use-kimi-api-to-complete-tool-calls#%E6%89%A7%E8%A1%8C%E5%B7%A5%E5%85%B7

    while finish_reason is None or finish_reason == "tool_calls":
        completion = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=messages,
            temperature=0.3,
            tools=tools,  # <-- 我们通过 tools 参数，将定义好的 tools 提交给 Kimi 大模型
        )
        choice = completion.choices[0]
        finish_reason = choice.finish_reason
        if finish_reason == "tool_calls":  # <-- 判断当前返回内容是否包含 tool_calls
            messages.append(choice.message)  # <-- 我们将 Kimi 大模型返回给我们的 assistant 消息也添加到上下文中，以便于下次请求时 Kimi 大模型能理解我们的诉求
            for tool_call in choice.message.tool_calls:  # <-- tool_calls 可能是多个，因此我们使用循环逐个执行
                tool_call_name = tool_call.function.name
                tool_call_arguments = json.loads(
                    tool_call.function.arguments)  # <-- arguments 是序列化后的 JSON Object，我们需要使用 json.loads 反序列化一下
                tool_function = tool_map[tool_call_name]  # <-- 通过 tool_map 快速找到需要执行哪个函数
                tool_result = tool_function(tool_call_arguments)
                print(tool_call_name, tool_result)

                # 使用函数执行结果构造一个 role=tool 的 message，以此来向模型展示工具调用的结果；
                # 注意，我们需要在 message 中提供 tool_call_id 和 name 字段，以便 Kimi 大模型
                # 能正确匹配到对应的 tool_call。
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call_name,
                    "content": str(tool_result),
                    # <-- 我们约定使用字符串格式向 Kimi 大模型提交工具调用结果，因此在这里使用 json.dumps 将执行结果序列化成字符串
                })

    print(choice.message.content)  # <-- 在这里，我们才将模型生成的回复返回给用户


if __name__ == "__main__":
    # demo_register_tools()
    demo_run_tools()
