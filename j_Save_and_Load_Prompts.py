import os

from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = PromptTemplate(
    template='''Translate this sentence from English to Chinese. \nSentence: {sentence}\nTranslation:''',
    input_variables=["sentence"],
)

prompt.save("resources/translation_prompt.json")

# Loading the prompt template:
load_prompt("resources/translation_prompt.json")
# Returns PromptTemplate()


# 试试中文
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

system_prompt = SystemMessagePromptTemplate.from_template(template)
# system_prompt.save("resources/ch_prompt.json")   # NotImplementedError

"""
Warning:
Please be aware that LangChain’s prompt saving may not work with all
types of prompt templates. To mitigate this, you can utilize the pickle
library or .txt files to read and write any prompts that LangChain does not
support.
正如书中说所，LangChain提供了多种PromptTemplate类，但并不是都支持sava()，很多都还没有实现或者根本没有定义save()接口。
书中给的解决方案是使用Python原生的pickle library将PromptTemplate实例序列化后保存到文件中。
另一种可行的方案是：在构建Prompt Template时，都使用PromptTemplate类，后续再用该类型的实例再去初始化其它类。
参考如下代码：
"""

prompt = PromptTemplate(template=template, input_variables=["principles", "context"])
prompt.save("resources/ch_prompt.json")  # 中文会Unicode编码后保存在json中
prompt = load_prompt("resources/ch_prompt.json")

system_prompt = SystemMessagePromptTemplate(prompt=prompt)
chat_prompt = ChatPromptTemplate.from_messages([system_prompt])

model = ChatOpenAI(
    base_url="https://api.moonshot.cn/v1",
    api_key=os.environ["MOONSHOT_API_KEY"],
    model="moonshot-v1-32k",
    max_tokens=20000
)

chain = chat_prompt | model

# 让Moonshot给星铁想个名字
result = chain.invoke({
    "industry": "游戏",
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
