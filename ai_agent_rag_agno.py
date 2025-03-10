from agno.agent import Agent
from agno.embedder.ollama import OllamaEmbedder
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.models.groq import Groq
from agno.vectordb.pgvector import PgVector, SearchType

from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
import numpy as np

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Create Groq model
model=Groq(id='qwen-qwq-32b')

# Create Ollama embedder
embedder = OllamaEmbedder(id="nomic-embed-text", dimensions=768)

# 保存原始方法
original_get_embedding = embedder.get_embedding

# 临时修复嵌入器输出
def fixed_get_embedding(text):
    embedding = original_get_embedding(text)
    return np.array(embedding).flatten().tolist()

embedder.get_embedding = fixed_get_embedding

# Create a knowledge base of PDFs from URLs
#knowledge_base = PDFUrlKnowledgeBase(
#    urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    # Use PgVector as the vector database and store embeddings in the `ai.recipes` table
#    vector_db=PgVector(
#        table_name="recipes",
#        db_url=db_url,
#        search_type=SearchType.hybrid,
#        embedder=embedder,
#    ),
#)

# Create a knowledge base of PDFs from local path
pdf_path = "D:/Chrome/Downloads/ThaiRecipes.pdf"

knowledge_base = PDFKnowledgeBase(
    path=pdf_path,
    # Use PgVector as the vector database and store embeddings in the `ai.recipes` table
    vector_db=PgVector(
        table_name="recipes",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=embedder,
    ),
)

# Load the knowledge base: Comment after first run as the knowledge base is already loaded
knowledge_base.load(recreate=True)

agent = Agent(
    model=model,
    tools=[DuckDuckGoTools(), Newspaper4kTools()],
    knowledge=knowledge_base,
    search_knowledge=True,
    show_tool_calls=True,
    markdown=True,
    add_datetime_to_instructions=True,
)

def update_knowledge_base(path):
    global knowledge_base  # 确保 `knowledge_base` 变量同步更新

    # 创建新的知识库并更新 `agent.knowledge`
    new_knowledge_base = PDFKnowledgeBase(
        path=path,
        vector_db=PgVector(
            table_name="recipes",
            db_url=db_url,
            search_type=SearchType.hybrid,
            embedder=embedder,
        ),
    )
    # 重新加载知识库
    new_knowledge_base.load(recreate=True)

    # 更新全局 `knowledge_base` 和 `agent.knowledge`
    knowledge_base = new_knowledge_base
    agent.knowledge = knowledge_base

    return knowledge_base


# 测试嵌入器输出
def test_fixed_get_embedding():
    print("测试嵌入器输出...")
    test_text = "Test sentence"
    test_embedding = embedder.get_embedding(test_text)  # 使用 get_embedding
    print(f"原始嵌入: {test_embedding[:5]}")  # 显示前5个值
    print(f"维度: {len(test_embedding)}")
    if isinstance(test_embedding, list) and all(isinstance(x, (int, float)) for x in test_embedding):
        print("嵌入是一维列表，格式正确")
    else:
        print("嵌入格式错误，尝试展平...")
        test_embedding = np.array(test_embedding).flatten().tolist()
        print(f"展平后嵌入: {test_embedding[:5]}, 维度: {len(test_embedding)}")


# 提供外部调用的 RAG 问答接口
def query_rag(question):
    """基于 PDF 知识库的 RAG 查询"""
    return agent.print_response(question, markdown=True,stream = True)

# 交互模式（仅在 CLI 下运行）
def agent_respond():
    print("🤖 T-AI RAG Agent：输入 'exit' 退出对话。")
    while True:
        print(f'\n')
        user_input = input("💬 请输入你的问题: ")
        if user_input.lower() in ["exit", "退出"]:
            print("🔚 结束对话。")
            break
        query_rag(user_input)

# 测试模式
if __name__ == "__main__":
    agent_respond()