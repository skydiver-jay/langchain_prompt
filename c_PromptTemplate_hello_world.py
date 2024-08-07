from langchain_core.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.chat_models.moonshot import MoonshotChat

prompt = PromptTemplate(
    template='''你是一个高级翻译助手，负责将{input_language}翻译为{output_language}.''',
    input_variables=["input_language", "output_language"]
)

system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)

chat = MoonshotChat()

# 书中的样例代码只设置了system prompt --> SystemMessage，所以如果需要执行翻译任务，还需要再构造一个HumanMessage，类似`a_moonshot_hello_world.py`中的例子
# result = chat.invoke(system_message_prompt.format_messages(input_language="中文", output_language="英文"))
# print(result.content)

print('~~~~~~~~~~~~~~以下为添加Human Message后的回答~~~~~~~~~~~~~~')

human_prompt = PromptTemplate(
    template='''请翻译如下内容：
            {content}。''',
    input_variables=["content"]
)

human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)

# 【注意】：经过测试，先后将SystemMessage和HumanMessage分别通过API调用LLM，不会得到续期效果，而是需要将两者组合成List一起传给LLM
# 由于SystemMessagePromptTemplate和HumanMessagePromptTemplate的format_messages()已经返回的是List[BaseMessage]类型，在一起构造一个List传给invoke()会出错
# 所以需要使用format()，只返回BaseMessage类似的实例
messages = [
    system_message_prompt.format(input_language="中文", output_language="英文"),
    human_message_prompt.format(content='''
            这片银河中有名为“星神”的存在，他们造就现实，抹消星辰，在无数“世界”中留下他们的痕迹。
            你——一名特殊的旅客，将与继承“开拓”意志的同伴一起，乘坐星穹列车穿越银河，沿着某位“星神”曾经所行之途前进。
            你将由此探索新的文明，结识新的伙伴，在无数光怪陆离的“世界”与“世界”之间展开新的冒险。所有你想知道的，都将在群星中找到答案。
            那么，准备好开始这段“开拓”之旅了吗？
            愿此行，终抵群星。'''),
]

result2 = chat.invoke(messages)

print(result2.content)


'''
MoonShot的回答：

In this galaxy, there exists a being known as the "Star Deity," who shapes reality, erases stars, and leaves their mark across countless "worlds."
You—a special traveler—will embark on a journey with companions who inherit the will to "pioneer," aboard the Star Dome train, traversing the galaxy along the path once taken by a certain "Star Deity."
You will explore new civilizations, meet new companions, and embark on new adventures between the myriad of bizarre "worlds." All that you wish to know will be found among the stars.
So, are you ready to begin this journey of "pioneering"?
May this voyage ultimately reach the stars...
'''