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

import streamlit as st
import tempfile

# Proxy Server（Clash/V2Ray/Trojan port）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# 初始化函数
def initialize_team():
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
        path="D:/Chrome/Downloads/ThaiRecipes.pdf",
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
        instructions="Always include sources",
        show_tool_calls=True,
        markdown=True,
    )

    finance_agent = Agent(
        name="Finance Agent",
        role="Get financial data",
        model=model,
        tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
        instructions="Use tables to display data",
        show_tool_calls=True,
        markdown=True,
    )

    email_agent = Agent(
        name="Email Agent",
        role="Responsible for Gmail email management",
        model=model,
        tools=[gmail_tools],
        instructions=[
            "You can use GmailTools to perform email operations",
            "You can read the latest emails, unread emails, search emails, send emails, create drafts, and reply to emails",
        ],
        show_tool_calls=True,
        markdown=True,
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
        ],
        knowledge=knowledge_base,  # 绑定组合知识库
        update_knowledge=False,    # 只查询，不更新
        search_knowledge=True,     # 启用知识库搜索
        show_tool_calls=True,
        markdown=True,
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
            "Ensure the steps are logically ordered and easy to follow."
        ],
        show_tool_calls=True,
        markdown=True,
        debug_mode=True,
        storage=storage,
        share_member_interactions=True,
        enable_team_history=True,
        num_of_interactions_from_history=5,
        add_datetime_to_instructions=True,
    )

    # 加载知识库（仅首次）
    if knowledge_agent.knowledge is not None:
        knowledge_agent.knowledge.load(recreate=False, upsert=True)

    return agent_team, knowledge_agent,local_pdf_knowledge_base


# 检查是否已初始化
# 初始化检查
if "agent_team" not in st.session_state or "knowledge_agent" not in st.session_state:
    agent_team, knowledge_agent,local_pdf_knowledge_base= initialize_team()
    st.session_state.agent_team = agent_team
    st.session_state.knowledge_agent = knowledge_agent
    st.session_state.local_pdf_knowledge_base = local_pdf_knowledge_base


# 使用持久化的 agent_team
agent_team = st.session_state.agent_team
knowledge_agent = st.session_state.knowledge_agent
local_pdf_knowledge_base = st.session_state.local_pdf_knowledge_base

# Streamlit UI

st.set_page_config(page_title="AI Team Chat", layout="wide")
st.title("🧠 AI Team 聊天系统")

# 会话状态初始化
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 左侧上传文件扩展知识库
with st.sidebar:
    st.header("📂 上传 PDF 添加至知识库")
    uploaded_files = st.file_uploader("拖入 PDF 文件", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            local_pdf_knowledge_base.path = tmp_path
            local_pdf_knowledge_base.load(recreate=False, upsert=True)
            knowledge_agent.knowledge.load(recreate=False, upsert=True)
            st.success(f"✅ {uploaded_file.name} 已加入知识库")

# 主界面聊天输入
user_input = st.chat_input("请输入问题或任务...")
chat_container = st.container()

# 展示历史聊天记录
for role, content in st.session_state.chat_history:
    with chat_container.chat_message(role):
        st.markdown(content)

# 处理新消息
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with chat_container.chat_message("user"):
        st.markdown(user_input)

    with chat_container.chat_message("assistant"):
        full_response = ""
        response_placeholder = st.empty()
        response_stream = agent_team.run(user_input, stream=True)

        for chunk in response_stream:
            if chunk.content:
                full_response += chunk.content
                response_placeholder.markdown(full_response)

        st.session_state.chat_history.append(("assistant", full_response))

