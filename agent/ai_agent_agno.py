from agno.agent import Agent
from agno.embedder.ollama import OllamaEmbedder
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.knowledge.website import WebsiteKnowledgeBase
from agno.storage.postgres import PostgresStorage
from agno.models.groq import Groq
from agno.vectordb.pgvector import PgVector, SearchType

from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.tools.yfinance import YFinanceTools
from tools.email_tool_agno import SendEmailTools
import numpy as np

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
storage = PostgresStorage(table_name="agent_sessions", db_url=db_url)

# Create Groq model
model=Groq(id='qwen-qwq-32b',timeout=60)

# Create Ollama embedder
embedder = OllamaEmbedder(id="nomic-embed-text", dimensions=768)

# Create a knowledge base of PDFs from URLs
#knowledge_base = PDFUrlKnowledgeBase(
#    urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    # Use PgVector as the vector database and store embeddings in the `ai.recipes` table
#    vector_db=PgVector(
#        table_name="pdf_url_documents",
#        db_url=db_url,
#        search_type=SearchType.hybrid,
#        embedder=embedder,
#    ),
#)

# Create a knowledge base of PDFs from local path
#pdf_path = "D:/Chrome/Downloads/ThaiRecipes.pdf"

#knowledge_base = PDFKnowledgeBase(
#    path=pdf_path,
#   # Use PgVector as the vector database and store embeddings in the `ai.recipes` table
#    vector_db=PgVector(
#        table_name="pdf_documents",
#        db_url=db_url,
#        search_type=SearchType.hybrid,
#        embedder=embedder,
#    ),
#)

# Create Website Knowledge Base
knowledge_base = WebsiteKnowledgeBase(
    urls=["https://docs.agno.com/introduction"],
    # Number of links to follow from the seed URLs
    max_links=10,
    #Maximum depth to crawl
    max_depth=5,
    # Table name: website_documents
    vector_db=PgVector(
        table_name="website_documents",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=embedder,
    ),
)

agent = Agent(
    model=model,
    tools=[DuckDuckGoTools(), Newspaper4kTools(),SendEmailTools(),
           YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    show_tool_calls=True,
    knowledge=knowledge_base,
    search_knowledge=True,
    storage=storage,
    add_history_to_messages=True,
    read_chat_history=True,
    markdown=True,
    add_datetime_to_instructions=True,
    #session_id='172eabdb-9427-4867-aba9-e8b3bc24da3c',
)

# Comment out after the knowledge base is loaded
if agent.knowledge is not None:
    agent.knowledge.load(recreate=True)


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
    #agent.knowledge = knowledge_base

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
def query_agent(question, knowledge_base_override=None):
    """基于 PDF 知识库的 RAG 查询"""

    # 如果 `app_rag.py` 传入了新的 `knowledge_base`，使用它
    #if knowledge_base_override:
       # agent.knowledge = knowledge_base_override

    try:
        agent.print_response(question, markdown=True, stream=True)
    except Exception as e:
        print(f"⚠️ 处理查询失败: {e}")
    # 返回仅包含 content 的生成器
    #return (response.content for response in agent.run(question, stream=True))


# 交互模式（仅在 CLI 下运行）
def agent_respond():
    print("🤖 T-AI RAG Agent：输入 'exit' 退出对话。")
    while True:
        print(f'\n')
        user_input = input("💬 请输入你的问题: ")
        if user_input.lower() in ["exit", "退出"]:
            print("🔚 结束对话。")
            break
        query_agent(user_input)
        #for chunk in response_stream:
        #    print(chunk, end="", flush=True)  # CLI 下流式打印

# 测试模式
if __name__ == "__main__":
    agent_respond()