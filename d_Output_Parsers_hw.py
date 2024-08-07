from langchain_core.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain.output_parsers import PydanticOutputParser
from pydantic.v1 import BaseModel, Field
from typing import List


class BusinessName(BaseModel):
    name: str = Field(description="新业务或新产品的名称。")
    rating_score: float = Field(description='''对名称的评分，最低0分，最高10分。''')


class BusinessNames(BaseModel):
    names: List[BusinessName] = Field(description='''一组新业务或新产品的名称。''')


# Set up a parser + inject instructions into the prompt template:
parser = PydanticOutputParser(pydantic_object=BusinessNames)

principles = '''
1. 每个名称必须简短，易于记忆
2. 每个名称必须朗朗上口
3. 每个名称必须独一无二，没有被其它公司所使用
4. 名称只包含中文
'''

model = MoonshotChat()
# model = MoonshotChat(model="moonshot-v1-128k")
template = '''
你是一个创意顾问，负责头脑风暴，设计新业务或者新产品的名称。
你必须遵守以下原则：
{principles}
以及{format_instructions}。
请设计3个名称，该业务属于{industry}领域，并且与如下描述相关：
{context}。
'''

context = '''
这片银河中有名为“星神”的存在，他们造就现实，抹消星辰，在无数“世界”中留下他们的痕迹。
你——一名特殊的旅客，将与继承“开拓”意志的同伴一起，乘坐星穹列车穿越银河，沿着某位“星神”曾经所行之途前进。
你将由此探索新的文明，结识新的伙伴，在无数光怪陆离的“世界”与“世界”之间展开新的冒险。所有你想知道的，都将在群星中找到答案。
那么，准备好开始这段“开拓”之旅了吗？
愿此行，终抵群星。
'''

system_message_prompt = SystemMessagePromptTemplate.from_template(template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

# chain = chat_prompt | model

# result = chain.invoke(
#     {
#         "principles": principles,
#         "industry": "游戏",
#         "context": context,
#         "format_instructions": parser.get_format_instructions()
#     }
# )
# print("原始结果：")
# print(result.content)
#
# print("parse后的结果：")
# print(parser.parse(result.content))
#
# print("""Warning:
# You should take care of edge cases as well as adding error handling
# statements, since LLM outputs might not always be in your desired
# format.
# 作者说得很对，就Moonshot而言，以上代码运行10次左右，错误就有3次，model返回的内容并非完全符合格式要求，导致出错""")


# 当然LCEL语法支持直接使用|运算符将model的输出直接传给parser
# 以下代码运行后，将直接得到上面parse后的结果，想省钱的话就可以不运行，知道用法就好
chain = chat_prompt | model | parser
result = chain.invoke(
    {
        "principles": principles,
        "industry": "游戏",
        "context": context,
        "format_instructions": parser.get_format_instructions()
    }
)
print(result)


"""
model=default

第一次回答
原始结果：
{
  "names": [
    {
      "name": "星穹旅者",
      "rating_score": 8
    },
    {
      "name": "星神轨迹",
      "rating_score": 9
    },
    {
      "name": "银河开拓纪",
      "rating_score": 7
    }
  ]
}
parse后的结果：
names=[BusinessName(name='星穹旅者', rating_score=8.0), BusinessName(name='星神轨迹', rating_score=9.0), BusinessName(name='银河开拓纪', rating_score=7.0)]

第二次回答
原始结果：
{
  "names": [
    {
      "name": "星穹旅者",
      "rating_score": 9
    },
    {
      "name": "银河开拓号",
      "rating_score": 8
    },
    {
      "name": "星神轨迹",
      "rating_score": 7
    }
  ]
}
parse后的结果：
names=[BusinessName(name='星穹旅者', rating_score=9.0), BusinessName(name='银河开拓号', rating_score=8.0), BusinessName(name='星神轨迹', rating_score=7.0)]
"""


"""
model="moonshot-v1-128k"

第一次回答
原始结果：
{
  "names": [
    {
      "name": "星穹旅者",
      "rating_score": 9
    },
    {
      "name": "银河开拓号",
      "rating_score": 8
    },
    {
      "name": "星神轨迹",
      "rating_score": 7
    }
  ]
}
parse后的结果：
names=[BusinessName(name='星穹旅者', rating_score=9.0), BusinessName(name='银河开拓号', rating_score=8.0), BusinessName(name='星神轨迹', rating_score=7.0)]


第二次回答
原始结果：
{
  "names": [
    {
      "name": "星穹旅者",
      "rating_score": 8
    },
    {
      "name": "银河开拓号",
      "rating_score": 9
    },
    {
      "name": "星神轨迹",
      "rating_score": 7
    }
  ]
}
parse后的结果：
names=[BusinessName(name='星穹旅者', rating_score=8.0), BusinessName(name='银河开拓号', rating_score=9.0), BusinessName(name='星神轨迹', rating_score=7.0)]
"""