import os

from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


class Article(BaseModel):
    """Identifying key points and contrarian views in an article."""

    points: str = Field(..., description="Key points from the article")
    contrarian_points: Optional[str] = Field(None, description="Any contrarian points acknowledged in the article")
    author: Optional[str] = Field(None, description="Author of the article")


_EXTRACTION_TEMPLATE = """Extract and save the relevant entities
mentioned \
in the following passage together with their properties.
If a property is not present and is not required in the function
parameters,
do not include it in the output."""

# Create a prompt telling the LLM to extract information:
prompt = ChatPromptTemplate.from_messages({("system", _EXTRACTION_TEMPLATE), ("user", "{input}")})

# model = MoonshotChat()
model = ChatOpenAI(
    base_url="https://api.moonshot.cn/v1",
    api_key=os.environ["MOONSHOT_API_KEY"],
    model="moonshot-v1-8k"
)

pydantic_schemas = [Article]

# Convert Pydantic objects to the appropriate schema:
tools = [convert_to_openai_tool(p) for p in pydantic_schemas]
print(tools)

# Give the model access to these tools:
model_with_tools = model.bind_tools(tools=tools)

# Create an end to end chain:
chain = prompt | model_with_tools

with open("resources/article-1.txt", 'r', encoding='utf-8') as file:
    article_content = file.read()

result = chain.invoke(
    {
        "input": article_content
    }
).tool_calls

print(result)
