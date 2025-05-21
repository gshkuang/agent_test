import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding

# 加载环境变量
load_dotenv()

# 1. 加载data/documents下的所有txt文档
data_dir = "./data/documents"
documents = SimpleDirectoryReader(data_dir).load_data()

# 2. 配置全局 LLM
Settings.llm = OpenAI(
    model="qwen-turbo",
    api_base=os.environ.get("DASHSCOPE_API_BASE"),
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-ada-002",
    api_base=os.environ.get("OPENAI_API_BASE"),
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# 3. 构建向量索引
index = VectorStoreIndex.from_documents(documents)

# 4. 创建查询引擎
query_engine = index.as_query_engine()

# 5. 进行RAG问答
while True:
    query = input("请输入你的问题（输入exit退出）：")
    if query.strip().lower() == "exit":
        break
    response = query_engine.query(query)
    print("RAG回答：", response)
