import os
from agno.agent import Agent
from agno.knowledge.combined import CombinedKnowledgeBase
from agno.vectordb.pgvector import PgVector
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.knowledge.website import WebsiteKnowledgeBase
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.document.reader.pdf_reader import PDFReader
from agno.embedder.ollama import OllamaEmbedder
from agno.vectordb.search import SearchType

from agno.models.groq import Groq

from agno.tools.duckduckgo import DuckDuckGoTools
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.tools.gmail import GmailTools
from agno.tools.yfinance import YFinanceTools

from agent.team import Team
from config.config import TOKEN_PATH, CREDENTIALS_PATH, DB_TEAM_PATH

# Proxy Server（Clash/V2Ray/Trojan port）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

task ="Check the latest news on PS5 and then send the collected information by email to 18340825516@163.com"
#task = "Tell me something about ThaiRecipes.pdf from knowledge base"

# Create a storage backend using the Sqlite database
storage = SqliteAgentStorage(
    table_name="agent_team_sessions",
    # db_file: Sqlite database file
    db_file=DB_TEAM_PATH,
    mode="team"
)

# Create Groq model
model=Groq(id='qwen-qwq-32b',timeout=60)


# Set GmailTools
gmail_tools=GmailTools(
        credentials_path=CREDENTIALS_PATH,
        token_path=TOKEN_PATH,
        get_latest_emails=True,
        get_emails_from_user=True,
        get_unread_emails=True,
        get_starred_emails=True,
        get_emails_by_context=True,
        get_emails_by_date=True,
        get_emails_by_thread=True,
        create_draft_email=True,
        send_email=True,
        send_email_reply=True,
        search_emails=True,
)

# 初始化嵌入器
embedder = OllamaEmbedder(id="nomic-embed-text", dimensions=768)

# 构建知识库
url_pdf_knowledge_base = PDFUrlKnowledgeBase(
    #urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector(
        table_name="pdf_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        search_type=SearchType.hybrid,
        embedder=embedder,
    ),
)

website_knowledge_base = WebsiteKnowledgeBase(
    #urls=["https://docs.agno.com/tools/toolkits/website"],
    max_depth=1,
    max_links=1,
    vector_db=PgVector(
        table_name="website_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        search_type=SearchType.hybrid,
        embedder=embedder,
    ),
)

local_pdf_knowledge_base = PDFKnowledgeBase(
    path="D:/Chrome/Downloads/test.pdf",
    reader=PDFReader(chunk=True),
    vector_db=PgVector(
        table_name="pdf_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        search_type=SearchType.hybrid,
        embedder=embedder,
    ),
)

knowledge_base = CombinedKnowledgeBase(
    sources=[url_pdf_knowledge_base, website_knowledge_base, local_pdf_knowledge_base],
    vector_db=PgVector(
        table_name="combined_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        search_type=SearchType.hybrid,
        embedder=embedder,
    ),
)


web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=model,
    tools=[DuckDuckGoTools()],
    instructions=[
        "Always include sources",
        "Do not include any escape characters in your response, keep the strings clean and unmodified.",
        "Ensure that no escape characters (e.g., '\\', '\n', '\r') are used in any of the strings. This is critical to avoid breaking the message formatting and causing potential errors in future responses."
    ],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=model,
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions="Use tables to display data",
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
)

email_agent = Agent(
    name="Email Agent",
    role="Responsible for Gmail email management",
    model=model,
    tools=[gmail_tools],
    instructions=[
        "You can use GmailTools to perform email operations",
        "You can read the latest emails, unread emails, search emails, send emails, create drafts, and reply to emails",
        "Do not include any escape characters in your response, keep the strings clean and unmodified.",
        "Ensure that no escape characters (e.g., '\\', '\n', '\r') are used in any of the strings. This is critical to avoid breaking the message formatting and causing potential errors in future responses."

    ],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
)

knowledge_agent = Agent(
    name="Knowledge Agent",
    role="Responsible for querying the knowledge base and returning relevant content",
    model=model,
    tools=[],
    instructions=[
        "You are responsible for querying and retrieving information from the knowledge base.",
        "Use the provided knowledge base to answer questions or provide detailed information.",
        "If the requested information is not in the knowledge base, inform the user and suggest expanding the knowledge base.",
        "Present information in a clear and structured manner, using markdown formatting.",
        "Do not include any escape characters in your response, keep the strings clean and unmodified.",
        "Ensure that no escape characters (e.g., '\\', '\n', '\r') are used in any of the strings. This is critical to avoid breaking the message formatting and causing potential errors in future responses."
    ],
    knowledge=knowledge_base,  # 绑定组合知识库
    update_knowledge=False,    # 只查询，不更新
    search_knowledge=True,     # 启用知识库搜索
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
)

agent_team = Team(
    members=[web_agent, finance_agent,email_agent,knowledge_agent],
    name="Multitasking Team",
    mode="coordinate",
    model=model,
    instructions=[
        "Analyze the user's input and break it down into simple steps based on natural language logic.",
        "Do not modify the user's input, simply divide it into distinct tasks as they are presented in the original request.",
        "Each task should correspond to one action mentioned in the user's input, and each action should be a single step.",
        "Ensure the steps are logically ordered and easy to follow.",
        "Do not include any escape characters in your response, keep the strings clean and unmodified.",
        "Ensure that no escape characters (e.g., '\\', '\n', '\r') are used in any of the strings. This is critical to avoid breaking the message formatting and causing potential errors in future responses."
    ],
    show_members_responses=True,
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
    storage=storage,
    share_member_interactions=True,
    enable_team_history=True,
    num_of_interactions_from_history=3,
    add_datetime_to_instructions=True,
)

if knowledge_agent.knowledge is not None:
    knowledge_agent.knowledge.load(recreate=False, upsert=True)

#agent_team.print_response(task, stream=True)

def get_team_components():
    return agent_team, knowledge_agent, local_pdf_knowledge_base

def respond():
    print("🤖 T-AIAgent：输入 'exit' 退出对话。")
    while True:
        print(f'\n')
        user_input = input("💬 请输入你的问题: ")
        if user_input.lower() in ["exit", "退出"]:
            print("🔚 结束对话。")
            break
        agent_team.print_response(user_input, stream=False)


# 测试模式
if __name__ == "__main__":
    respond()