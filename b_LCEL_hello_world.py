"""
LangChain Expression Language (LCEL)

"""
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate

template = '''
你是一个创意顾问，负责头脑风暴，设计新业务或者新产品的名称。
你必须遵守以下原则：
{principles}
请设计3个名称，该业务属于{industry}领域，并且与如下描述相关：
{context}
以下是输出的格式：
1. 名称1
2. 名称2
3. 名称3
'''

# 初始化一个ChatModel实例
# model = MoonshotChat()

# 由于Moonshot的API接口与OpenAI的完全兼容，也可以使用ChatOpenAI来调用Moonshot AI
from langchain_openai import ChatOpenAI
import os
model = ChatOpenAI(
    base_url="https://api.moonshot.cn/v1",
    api_key=os.environ["MOONSHOT_API_KEY"],
    model="moonshot-v1-8k"
)

# 实例化一个SystemMessagePromptTemplate
system_prompt = SystemMessagePromptTemplate.from_template(template)

# 实例化一个ChatPromptTemplate，可见一个chat prompt是一个system prompt列表
chat_prompt = ChatPromptTemplate.from_messages([system_prompt])

# 构造一个Chain
chain = chat_prompt | model

# 让Moonshot给星铁想个名字
result = chain.invoke({
    "industry": "游戏",
    # 文案来自：https://baike.baidu.com/item/%E5%B4%A9%E5%9D%8F%EF%BC%9A%E6%98%9F%E7%A9%B9%E9%93%81%E9%81%93/58766453，后面会多次用到
    "context": '''
        这片银河中有名为“星神”的存在，他们造就现实，抹消星辰，在无数“世界”中留下他们的痕迹。
        你——一名特殊的旅客，将与继承“开拓”意志的同伴一起，乘坐星穹列车穿越银河，沿着某位“星神”曾经所行之途前进。
        你将由此探索新的文明，结识新的伙伴，在无数光怪陆离的“世界”与“世界”之间展开新的冒险。所有你想知道的，都将在群星中找到答案。
        那么，准备好开始这段“开拓”之旅了吗？
        愿此行，终抵群星。''',
    "principles": '''
        1. 每个名称必须简短，易于记忆。 
        2. 每个名称必须朗朗上口。
        3. 每个名称必须独一无二，没有被其它公司所使用. 
        4. 名称只包含中文'''
})

print(result.content)

"""
MoonShot的回答：

1. 星轨旅者
2. 银河开拓号
3. 星辰旅航团

1. 星轨旅者
2. 星神征途
3. 穹宇探索者
"""

