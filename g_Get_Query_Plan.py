import os
from langchain_openai.chat_models import ChatOpenAI
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate
from pydantic.v1 import BaseModel, Field
from typing import List


# Defining QueryPlan and Query allows you to first ask an LLM to parse a user’s query into multiple steps.
class Query(BaseModel):
    id: int
    question: str
    dependencies: List[int] = Field(default_factory=list,
                                    description="""A list of sub-queries that must be completed before this task can 
                                    be completed. Use a sub query when anything is unknown and we might need to ask 
                                    many queries to get an answer. Dependencies must only be other queries.""")


class QueryPlan(BaseModel):
    query_graph: List[Query]


# Set up a model:
llm = ChatOpenAI(
    base_url="https://api.moonshot.cn/v1",
    api_key=os.environ["MOONSHOT_API_KEY"],
    model="moonshot-v1-32k",
    max_tokens=20000
)

# Set up a parser:
parser = PydanticOutputParser(pydantic_object=QueryPlan)

template = """Generate a query plan. This will be used for task execution. Answer the following query: {query}.
Return the following query graph format: {format_instructions}"""

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

# Create the LCEL chain with the prompt, model, and parser:
# chain = chat_prompt | llm | parser
#
# result = chain.invoke(
#     {
#         "query": '''I want to get the results from my database. Then I want to find out what the average age of my
#         top 10 customers is. Once I have the average age, I want to send an email to John. Also I just generally want
#         to send a welcome introduction email to Sarah, regardless of the other tasks.''',
#         "format_instructions": parser.get_format_instructions()
#     }
# )
# print(result)
# print(result.query_graph)

'''
MoonShot对英文prompt的回答（parser后）: 
[
Query(id=1, question='Get results from the database', dependencies=[]), 
Query(id=2, question='Find the average age of the top 10 customers', dependencies=[1]), 
Query(id=3, question='Send an email to John with the average age', dependencies=[2]), 
Query(id=4, question='Send a welcome introduction email to Sarah', dependencies=[])
]
'''


# 试试换成中文
# 经过反复的调试，才使得kimi在中文prompt上获得与英文prompt相似的结果，但中文prompt相比英文不再那么的口语化，需要更明确、详细、特定格式的说明。并不是直接将英文prompt直译为中文就行。
# 注意点：
#     1. 对使用代码定义的class中的成员需要更多、更准确的描述（description）
#     2. 当prompt中包含中英文时（尤其是使用PydanticOutputParser生成格式说明时），最好明确告诉LLM使用中文回答问题。“请注意，上下文中可能存在中文外的其它语言，但最终请使用中文回答。”
#     3. 中文prompt中，**包裹的内容更容易被LLM识别为重要需求，比如**任务描述**、**格说明求**，如果没有这类关键字，而只是“请生成完成如下任务的任务序列: {query}”，很有可能无法获得预期结果。
#        应对任何重要需求是都这样构建prompt会比较好：
#            请生成完成如下任务的任务序列。
#            **任务描述**: {query}
#     4. 需要对LLM返回结果进行parser时，建议在prompt中增加：“只返回符合格说明求的内容，不要返回描述内容”。
class QueryCh(BaseModel):
    id: int = Field(description="本任务的id")
    question: str = Field(description="一项任务")
    dependencies: List[int] = Field(default_factory=list,
                                    description="""这是一组完成本任务所依赖的子任务id的列表。因为，当完成本任务遇到未知事项时，需要完成某个子任务才能获得信息；
                                    并且，可能需要完成多个子任务以获得此任务的最终结果。所依赖的子任务项必须只能是其它任务，不能依赖自身。""")


class QueryPlanCh(BaseModel):
    query_graph: List[QueryCh] = Field(description="任务列表")


parser = PydanticOutputParser(pydantic_object=QueryPlanCh)

template = """
请生成完成如下任务的任务序列。
**任务描述**: {query}
返回的任务序列格式请遵从如下格式要求。
**格说明求**: {format_instructions}。
另外，
1. 请注意，上下文中可能存在中文外的其它语言，但最终请使用中文回答。
2. 只返回符合格说明求的内容，不要返回描述内容。"""

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
chat_prompt_ch = ChatPromptTemplate.from_messages([system_message_prompt])

chain = chat_prompt_ch | llm | parser
# chain = chat_prompt_ch | llm

result_ch = chain.invoke(
    {
        "query": '''我打算从我的数据库中获得数据并分析。 我想分析得出我的Top 10的客户的平均年龄. 一旦有了分析结果，我需要把结果通过邮件发送给布洛妮娅。 
        另外，与以上事项无关，我还打算发送一封欢迎邮件给到希露瓦。''',
        "format_instructions": parser.get_format_instructions()
    }
)
print(result_ch)
print(result_ch.query_graph)


"""
[
QueryCh(id=1, question='从数据库中获取数据', dependencies=[]), 
QueryCh(id=2, question='分析Top 10客户的平均年龄', dependencies=[1]), 
QueryCh(id=3, question='通过邮件发送分析结果给布洛妮娅', dependencies=[2]), 
QueryCh(id=4, question='发送欢迎邮件给希露瓦', dependencies=[])
]
"""
