from langchain_community.llms import QianfanLLMEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Tongyi
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langchain.output_parsers import (
    PydanticOutputParser,
)  # 导入LangChain的PydanticOutputParser，用于解析模型输出
# 导入记忆组件
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# 加载环境变量
load_dotenv()


# 定义Pydantic模型
class GenerativeAIExample(BaseModel):
    """生成式AI的示例应用场景"""

    name: str = Field(..., description="应用名称")
    description: str = Field(..., description="应用描述")
    industry: str = Field(..., description="适用行业，如教育、医疗、金融等")


class GenerativeAIResponse(BaseModel):
    """生成式AI的回答模型"""

    definition: str = Field(..., description="生成式AI的定义")
    key_features: List[str] = Field(..., description="关键特性列表")
    examples: List[GenerativeAIExample] = Field(..., description="应用示例")


# 初始化通义千问模型
llm = Tongyi(model_name="qwen-turbo")

# 设置解析器
parser = PydanticOutputParser(pydantic_object=GenerativeAIResponse)

# 创建提示模板 - 使用普通的PromptTemplate而不是ChatPromptTemplate
prompt_template = PromptTemplate(
    template="请以JSON格式回答用户的问题：{query}\n\n{format_instructions}",
    input_variables=["query"],
    # partial_variables={"format_instructions": parser.get_format_instructions()},
)
prompt_template = prompt_template.partial(
    format_instructions=parser.get_format_instructions()
)
# 创建链
chain = prompt_template | llm | parser  # LLMChain(llm=llm, prompt=prompt_template)

# 调用链并获取结构化响应
print("\n结构化JSON响应示例:")
try:
    # 获取原始响应
    ai_info: GenerativeAIResponse = chain.invoke({"query": "什么是生成式AI？"})

    # 手动解析JSON

    print(f"响应: {ai_info}")
    # ai_info = parser.parse(response["text"])

    # 输出结构化数据
    print(f"定义: {ai_info.definition}")
    print("\n关键特性:")
    for feature in ai_info.key_features:
        print(f"- {feature}")

    print("\n应用示例:")
    for example in ai_info.examples:
        print(f"- {example.name} ({example.industry})")
        print(f"  {example.description}")

    # 将Pydantic模型转换回JSON
    print("\n转换回JSON:")
    print(ai_info.model_dump())

except Exception as e:
    print(f"处理错误: {e}")

# 添加带记忆功能的对话链示例
print("\n带记忆功能的对话示例:")

# 创建记忆组件
memory = ConversationBufferMemory()

# 创建带记忆的对话链
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True  # 设置为True可以查看链的执行过程
)

# 第一轮对话
response1 = conversation.predict(input="你好，我想了解一下生成式AI")
print(f"第一轮回答: {response1}")

# 第二轮对话 - 使用代词，测试记忆功能
response2 = conversation.predict(input="它有哪些主要应用场景？")
print(f"第二轮回答: {response2}")

# 查看记忆中存储的内容
print("\n记忆中的对话历史:")
print(memory.buffer)

