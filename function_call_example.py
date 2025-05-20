from langchain_community.llms import Tongyi
from langchain.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 初始化通义千问模型
llm = Tongyi(model_name="qwen-turbo")
print(llm("你好"))


# 定义工具函数
def search_web(query: str) -> str:
    """搜索网络获取信息

    Args:
        query: 搜索查询词

    Returns:
        搜索结果摘要
    """
    # 这里是模拟的搜索结果，实际应用中可以接入真实的搜索API
    search_results = {
        "人工智能": "人工智能（AI）是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。它包括机器学习、深度学习、自然语言处理等领域。",
        "机器学习": "机器学习是人工智能的一个子领域，专注于开发能够从数据中学习并做出预测的算法和模型，而无需显式编程。",
        "深度学习": "深度学习是机器学习的一个分支，使用多层神经网络处理复杂数据。它在图像识别、语音识别和自然语言处理等领域取得了突破性进展。",
        "大语言模型": "大语言模型（LLM）是基于Transformer架构的深度学习模型，通过大规模预训练能够理解和生成人类语言。代表模型包括GPT、LLaMA等。",
    }

    # 查找最匹配的关键词
    for key in search_results:
        if key in query:
            return search_results[key]

    return f"抱歉，没有找到关于'{query}'的搜索结果。"


def get_weather(location: str) -> str:
    """获取指定地点的天气信息

    Args:
        location: 地点名称，如"北京"、"上海"等

    Returns:
        天气信息
    """
    # 模拟的天气数据
    weather_data = {
        "北京": "晴天，温度25°C",
        "上海": "多云，温度28°C",
        "广州": "小雨，温度30°C",
        "深圳": "阵雨，温度29°C",
    }

    return weather_data.get(location, f"无法获取{location}的天气信息")


def calculate(expression: str) -> str:
    """计算数学表达式

    Args:
        expression: 数学表达式，如"2+2"

    Returns:
        计算结果
    """
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


# 创建工具列表
tools = [
    Tool(
        name="搜索",
        func=search_web,
        description="当你需要查找信息时使用此工具。输入搜索查询词。",
    ),
    Tool(
        name="天气查询",
        func=get_weather,
        description="当你需要获取某地天气信息时使用此工具。输入地点名称。",
    ),
    Tool(
        name="计算器",
        func=calculate,
        description="当你需要计算数学表达式时使用此工具。输入数学表达式。",
    ),
]

# 创建提示模板
prompt = PromptTemplate.from_template(
    """你是一个智能助手，可以回答用户的问题。
    你可以使用以下工具来帮助回答问题：

    {{tools}}

    使用工具时，请按照以下格式：
    思考：思考你应该如何回答问题，以及是否需要使用工具
    行动：工具名称，参数
    观察：工具返回的结果
    ...（可以有多轮思考-行动-观察）
    回答：根据所有信息给出最终回答

    {agent_scratchpad}

    用户问题：{input}
    """
)

# 创建基于OpenAI函数调用的代理
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def run_examples():
    """运行函数调用示例"""
    print("\n================ 函数调用示例 ================")

    # 示例1：搜索信息
    query1 = "什么是人工智能？"
    print(f"\n用户: {query1}")
    response1 = agent_executor.invoke({"input": query1})
    print(f"助手: {response1}")

    # 示例2：天气查询
    query2 = "北京今天天气怎么样？"
    print(f"\n用户: {query2}")
    response2 = agent_executor.invoke({"input": query2})
    print(f"助手: {response2['output']}")

    # 示例3：计算
    query3 = "计算23乘以45"
    print(f"\n用户: {query3}")
    response3 = agent_executor.invoke({"input": query3})
    print(f"助手: {response3['output']}")

    # 示例4：混合查询
    query4 = "深圳的天气如何？另外，告诉我什么是大语言模型。"
    print(f"\n用户: {query4}")
    response4 = agent_executor.invoke({"input": query4})
    print(f"助手: {response4['output']}")


if __name__ == "__main__":
    run_examples()
