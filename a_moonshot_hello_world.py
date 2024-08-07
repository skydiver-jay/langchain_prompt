import os
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.messages import HumanMessage, SystemMessage

# Generate your api key from: https://platform.moonshot.cn/console/api-keys
# os.environ["MOONSHOT_API_KEY"] = "MOONSHOT_API_KEY"

chat = MoonshotChat()
# or use a specific model
# Available models: https://platform.moonshot.cn/docs
# chat = MoonshotChat(model="moonshot-v1-128k")
# 默认使用的是moonshot-v1-8k


messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to Chinese."
    ),
    HumanMessage(
        content="Translate this sentence from English to Chinese. "
                "You are a helpful assistant that translates English to Chinese."
    ),
]

print(chat.invoke(messages))


"""
Moonshot收费标准

模型	计费单位	价格
moonshot-v1-8k	    1M tokens	¥12.00
moonshot-v1-32k	    1M tokens	¥24.00
moonshot-v1-128k	1M tokens	¥60.00
此处 1M = 1,000,000，表格中的价格代表每消耗 1M tokens 的价格。

以上模型的区别在于它们的最大上下文长度，这个长度包括了输入消息和生成的输出，在效果上并没有什么区别。

收费接口说明

Chat Completion 接口收费：按照实际输入输出 tokens 的消耗计费
文件相关接口（文件内容抽取/文件存储）接口限时免费
"""