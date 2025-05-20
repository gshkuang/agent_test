from langchain_community.llms import Tongyi
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence, Literal, Union
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
import operator

# 加载环境变量
load_dotenv()

# 初始化通义千问模型
llm = Tongyi(model_name="qwen-turbo")

# ===================== LangGraph示例 =====================
print("\n\n================ LangGraph示例 ================")

# 定义状态类型
class AgentState(TypedDict):
    messages: Annotated[Sequence[Union[HumanMessage, AIMessage]], operator.add]
    next: str

# 定义工具函数
def search_weather(location: str) -> str:
    """模拟天气查询工具"""
    weather_data = {
        "北京": "晴天，温度25°C",
        "上海": "多云，温度28°C",
        "广州": "小雨，温度30°C",
        "深圳": "阵雨，温度29°C"
    }
    return weather_data.get(location, f"无法获取{location}的天气信息")

def calculate(expression: str) -> str:
    """简单计算器工具"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

# 定义路由函数
def router(state: AgentState) -> Literal["search_weather", "calculate", "respond"]:
    """根据用户输入决定下一步操作"""
    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        return "respond"
    
    query = last_message.content.lower()
    
    if "天气" in query or "温度" in query:
        return "search_weather"
    elif any(op in query for op in ["+", "-", "*", "/", "计算"]):
        return "calculate"
    else:
        return "respond"

# 定义天气查询节点
def search_weather_node(state: AgentState) -> AgentState:
    """处理天气查询请求"""
    last_message = state["messages"][-1]
    query = last_message.content
    
    # 提取地点
    location_prompt = PromptTemplate(
        template="从以下查询中提取需要查询天气的地点名称，只返回地点名称：{query}",
        input_variables=["query"]
    )
    location_chain = LLMChain(llm=llm, prompt=location_prompt)
    location = location_chain.run(query=query).strip()
    
    # 查询天气
    weather_info = search_weather(location)
    
    # 添加结果到消息
    new_message = AIMessage(content=f"我查询到{location}的天气信息：{weather_info}")
    return {"messages": state["messages"] + [new_message], "next": ""}

# 定义计算节点
def calculate_node(state: AgentState) -> AgentState:
    """处理计算请求"""
    last_message = state["messages"][-1]
    query = last_message.content
    
    # 提取表达式
    expression_prompt = PromptTemplate(
        template="从以下查询中提取数学表达式，只返回可以直接计算的表达式（如：1+2，3*4等）：{query}",
        input_variables=["query"]
    )
    expression_chain = LLMChain(llm=llm, prompt=expression_prompt)
    expression = expression_chain.run(query=query).strip()
    
    # 计算结果
    result = calculate(expression)
    
    # 添加结果到消息
    new_message = AIMessage(content=f"计算表达式 '{expression}' 的结果是：{result}")
    return {"messages": state["messages"] + [new_message], "next": ""}

# 定义响应节点
def respond(state: AgentState) -> AgentState:
    """生成一般性回复"""
    messages = state["messages"]
    response = llm.invoke(messages[-1].content)
    new_message = AIMessage(content=response)
    return {"messages": messages + [new_message], "next": ""}

# Define a simple start node (does nothing, just passes state)
def start_node(state: AgentState) -> AgentState:
    return state

# 创建图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("start", start_node) # Add the start node
workflow.add_node("search_weather", search_weather_node)
workflow.add_node("calculate", calculate_node)
workflow.add_node("respond", respond)

# 设置入口点
workflow.set_entry_point("start") # Set start node as entry point

# 设置边
workflow.add_conditional_edges(
    "start", # Source node is "start"
    router,   # Conditional function is the router function
    {
        "search_weather": "search_weather",
        "calculate": "calculate",
        "respond": "respond"
    }
)
workflow.add_edge("search_weather", END)
workflow.add_edge("calculate", END)
workflow.add_edge("respond", END)

# 编译图
agent = workflow.compile()

def run_examples():
    """运行LangGraph示例"""
    # 测试LangGraph
    print("\nLangGraph对话代理测试:")

    # 测试天气查询
    messages = [HumanMessage(content="北京今天的天气怎么样？")]
    result = agent.invoke({"messages": messages, "next": ""})
    print("\n用户: 北京今天的天气怎么样？")
    print(f"代理: {result['messages'][-1].content}")

    # 测试计算功能
    messages = [HumanMessage(content="帮我计算一下23乘以45等于多少")]
    result = agent.invoke({"messages": messages, "next": ""})
    print("\n用户: 帮我计算一下23乘以45等于多少")
    print(f"代理: {result['messages'][-1].content}")

    # 测试一般性问题
    messages = [HumanMessage(content="什么是人工智能？")]
    result = agent.invoke({"messages": messages, "next": ""})
    print("\n用户: 什么是人工智能？")
    print(f"代理: {result['messages'][-1].content}")

if __name__ == "__main__":
    run_examples()