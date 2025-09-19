import os

from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.team import Team
from agno.tools.dalle import DalleTools
from agno.tools.file import FileTools
from agno.vectordb.pgvector import PgVector, SearchType

from agno.knowledge.embedder.openai import OpenAIEmbedder

from agno.models.groq import Groq
from agno.models.openai import OpenAIChat

from agno.tools.duckduckgo import DuckDuckGoTools
from agno.db.sqlite.sqlite import SqliteDb
from agno.tools.gmail import GmailTools
from agno.tools.yfinance import YFinanceTools

from config.config import TOKEN_PATH, CREDENTIALS_PATH, DB_TEAM_PATH, DOWNLOAD_DIR

# Proxy Server（Clash/V2Ray/Trojan port）
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

task ="Check the latest news on PS5 and then send the collected information by email to 18340825516@163.com"
#task = "Tell me something about ThaiRecipes.pdf from knowledge base"

# Set GmailTools
gmail_tools=GmailTools(
        credentials_path=CREDENTIALS_PATH,
        token_path=TOKEN_PATH,
        include_tools=["get_latest_emails","send_email"],
)

# Create a storage backend using the Sqlite database
storage = SqliteDb(
    # db_file: Sqlite database file
    db_file=DB_TEAM_PATH,
)

# Create Groq model
model=Groq(id='qwen-qwq-32b',timeout=60)

# Create embedder
embedder = OpenAIEmbedder()

# Create shared vector db
shared_vector_db = PgVector(
    table_name="documents",
    db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    search_type=SearchType.hybrid,
    embedder=OpenAIEmbedder(),
)

# Create knowledgebase
knowledge_base = Knowledge(
    vector_db=shared_vector_db
)

# Add content to knowledge base
knowledge_base.add_content(url="https://docs.agno.com/llms-full.txt")

# Create web search agent for supplementary information
web_agent = Agent(
    name="Web Search Agent",
    role="Handle web search requests",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[DuckDuckGoTools()],
    instructions=["Always include sources"],
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=OpenAIChat("gpt-4o-mini"),
    tools=[YFinanceTools()],
    instructions="Use tables to display data",
    markdown=True,
    debug_mode=True,
)

email_agent = Agent(
    name="Email Agent",
    role="Responsible for Gmail email management",
    model=OpenAIChat("gpt-4o-mini"),
    tools=[gmail_tools],
    instructions=[
        "You can use GmailTools to perform email operations",
        "You can read the latest emails, unread emails, search emails, send emails, create drafts, and reply to emails",
    ],
    markdown=True,
    debug_mode=True,
)

file_agent = Agent(
    name="File Agent",
    role="Responsible for local files management",
    model=OpenAIChat("gpt-4o"),
    tools=[FileTools(base_dir=DOWNLOAD_DIR),],
    instructions=["You can use FileTools to read and write files on the local file system",],
    markdown=True,
    debug_mode= True,
)

image_agent = Agent(
    name="Image Agent",
    role="Responsible for generating images using DALL-E",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DalleTools()],
    description="You are an AI agent that can generate images using DALL-E.",
    instructions="When the user asks you to create an image, use the `create_image` tool to create the image.",
    markdown=True,
)


agent_team = Team(
    name="Multitasking Team",
    members=[web_agent, finance_agent,email_agent,file_agent,image_agent],
    model=OpenAIChat(id="gpt-4o"),
    knowledge=knowledge_base,
    show_members_responses=True,
    markdown=True,
    debug_mode=True,
    share_member_interactions=True,
)

if __name__ == "__main__":
    agent_team.print_response("Tell me about the Agno framework", stream=True)