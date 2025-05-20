import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Tongyi
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor

# 加载环境变量
load_dotenv()


def create_vector_db():
    """创建向量数据库"""
    # 加载文档
    loader = TextLoader("./data/documents/sample.txt")
    documents = loader.load()

    # 分割文档
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # 创建向量存储
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
    vector_store = FAISS.from_documents(texts, embeddings)

    return vector_store


def create_rag_agent():
    """创建具有RAG功能的Agent"""
    # 创建向量数据库
    vector_store = create_vector_db()

    # 创建检索器
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # 创建语言模型
    llm = Tongyi(model_name="qwen-max")

    # 创建RAG检索QA链
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。
    
    {context}
    
    问题: {question}
    回答: """

    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
    )

    # 定义工具
    tools = [
        Tool(
            name="RAG检索问答系统",
            func=rag_chain.run,
            description="当你需要回答关于人工智能、深度学习、大语言模型或RAG等知识的问题时，使用这个工具。",
        )
    ]

    # 创建agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    return agent


def main():
    """主函数"""
    print("正在初始化RAG Agent...")
    agent = create_rag_agent()

    print("\n初始化完成！现在您可以向Agent提问。输入'退出'结束对话。")

    while True:
        user_input = input("\n您的问题: ")
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("感谢使用！再见！")
            break

        try:
            response = agent.run(user_input)  # 用 run 即可
            print(f"Agent回答: {response}")
        except Exception as e:
            print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()
