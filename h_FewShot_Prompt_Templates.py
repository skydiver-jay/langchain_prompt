import os

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

examples = [
    {
        "question": "What is the capital of France?",
        "answer": "Paris",
    },
    {
        "question": "What is the capital of Spain?",
        "answer": "Madrid",
    }  # ...more examples...
]

# example模板中的参数question和answer 是和 上面定义的examples中的key相对应
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{question}"),
        ("ai", "{answer}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

print(few_shot_prompt.format())

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", '''You are responsible for answering questions about countries. Only return the country name.'''),
        few_shot_prompt,
        ("human", "{question}"),
    ]
)

model = ChatOpenAI(
    base_url="https://api.moonshot.cn/v1",
    api_key=os.environ["MOONSHOT_API_KEY"],
    model="moonshot-v1-32k",
    max_tokens=20000
)

# Creating the LCEL chain with the prompt, model, and a StrOutputParser():
# chain = final_prompt | model | StrOutputParser()
#
# result = chain.invoke({"question": "What is the capital of Taiwan?"})   # Taipei， kimi也会翻车呀
#
# print(result)


# 换成中文试试
examples_ch = [
    {
        "question": "法国的首都是哪里?",
        "answer": "巴黎",
    },
    {
        "question": "西班牙的首都是哪里?",
        "answer": "马德里",
    }  # ...more examples...
]

few_shot_prompt_ch = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples_ch,
)
print(few_shot_prompt_ch.format())

final_prompt_ch = ChatPromptTemplate.from_messages(
    [
        ("system", '''你负责回答关于国家的问题'''),
        few_shot_prompt_ch,
        ("human", "{question}"),
    ]
)

chain = few_shot_prompt_ch | model | StrOutputParser()

result = chain.invoke({"question": "台湾的首都是哪里?"})   # kimi很懂的，就不回答了

print(result)


'''
政治问题还是留给国家去解决，对软件产品各个角度都无死角的提要求，不切实际，也没什么意义，这方面投入太多，开心的只是对手和敌人
'''