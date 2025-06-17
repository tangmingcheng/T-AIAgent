import os

from agno.agent import Agent
from agno.knowledge.combined import CombinedKnowledgeBase
from agno.tools.dalle import DalleTools
from agno.tools.file import FileTools
from agno.vectordb.pgvector import PgVector
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.knowledge.website import WebsiteKnowledgeBase
from knowledge.pdf import PDFKnowledgeBase
from agno.document.reader.pdf_reader import PDFReader
from agno.embedder.openai import OpenAIEmbedder
from agno.vectordb.search import SearchType

from agno.models.groq import Groq
from agno.models.openai import OpenAIChat

from agno.tools.duckduckgo import DuckDuckGoTools
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.tools.gmail import GmailTools
from agno.tools.yfinance import YFinanceTools

from agno.team import Team
from config.config import TOKEN_PATH, CREDENTIALS_PATH, DB_TEAM_PATH, DOWNLOAD_DIR

# Proxy Server（Clash/V2Ray/Trojan port）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

task ="Check the latest news on PS5 and then send the collected information by email to 18340825516@163.com"
#task = "Tell me something about ThaiRecipes.pdf from knowledge base"

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

# Create a storage backend using the Sqlite database
storage = SqliteAgentStorage(
    table_name="agent_team_sessions",
    # db_file: Sqlite database file
    db_file=DB_TEAM_PATH,
    mode="team"
)

# Create Groq model
model=Groq(id='qwen-qwq-32b',timeout=60)

# Create embedder
embedder = OpenAIEmbedder()

# Create shared vector db
shared_vector_db = PgVector(
    table_name="combined_documents",
    db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    search_type=SearchType.hybrid,
    embedder=OpenAIEmbedder(),
)

# Create knowledgebase
url_pdf_knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=shared_vector_db,
)

website_knowledge_base = WebsiteKnowledgeBase(
    #urls=["https://docs.agno.com/tools/toolkits/website"],
    max_depth=1,
    max_links=1,
    vector_db=shared_vector_db,
)

local_pdf_knowledge_base = PDFKnowledgeBase(
    #path="D:/Chrome/Downloads/ThaiRecipes.pdf",
    reader=PDFReader(),
    vector_db=shared_vector_db,
)

knowledge_base = CombinedKnowledgeBase(
    sources=[url_pdf_knowledge_base, website_knowledge_base, local_pdf_knowledge_base],
    vector_db=shared_vector_db
)

web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=model,
    tools=[DuckDuckGoTools()],
    instructions=["Always include sources",],
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
    ],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
)

knowledge_agent = Agent(
    name="Knowledge Agent",
    role="Responsible for querying the knowledge base and returning relevant content",
    model=OpenAIChat("gpt-4o-mini"),
    instructions=[
        "You are responsible for querying and retrieving information from the knowledge base.",
        "Use the provided knowledge base to answer questions or provide detailed information.",
        "If the requested information is not in the knowledge base, inform the user and suggest expanding the knowledge base.",
        "Present information in a clear and structured manner, using markdown formatting.",
    ],
    knowledge=knowledge_base,
    update_knowledge=False,
    search_knowledge=True,
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
)

file_agent = Agent(
    name="File Agent",
    role="Responsible for local files management",
    model=OpenAIChat("gpt-4o"),
    tools=[FileTools(base_dir=DOWNLOAD_DIR),],
    instructions=["You can use FileTools to read and write files on the local file system",],
    show_tool_calls=True,
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
    show_tool_calls=True,
)

agent_team = Team(
    members=[web_agent, finance_agent,email_agent,knowledge_agent,file_agent,image_agent],
    name="Multitasking Team",
    mode="coordinate",
    model=OpenAIChat(id="gpt-4o"),
    description="You are a smart team coordinator capable of understanding user instructions and delegating tasks.",
    instructions=[
        "Your role is to act as the leader of a team of specialized agents. When a task is received, analyze the user's instruction and decompose it into smaller, meaningful subtasks if necessary.",
        "Assign each subtask to the most appropriate agent based on its domain of expertise.",
        "If no agent is suitable for a specific subtask, you must handle it yourself.",
        "You are allowed to process tasks sequentially or in parallel, depending on the logical order and dependency between subtasks.",
        "Ensure that the task flow remains coherent and follows the intended execution order.",
        "You may coordinate multiple agents in one session and must integrate their results to deliver a final, high-quality response.",
        "Always return your final answer in a clear, structured, and helpful way using markdown format.",
        "You should remember the last few turns of conversation. If the user refers to something mentioned earlier (such as a generated output or an incomplete instruction), infer and resolve it based on recent context.",
        "Do not ask the user to repeat what was said earlier unless the meaning is truly ambiguous or multiple interpretations exist.",
        "When relying on prior context, be cautious and validate that your assumption is logically consistent with the user's intent.",
        "Only carry context over when it is semantically meaningful and safe to do so.",
        "When a user query involves concepts such as 'knowledge base', 'pdf', 'website content', or refers to past learned information, you must prioritize assigning the task to the knowledge_agent, as it is responsible for vector search and semantic document retrieval.",
        "When you determine that a subtask should be handled by a specific agent, you must assign it to that agent explicitly. **You are not allowed to use that agent’s tools yourself**, as you do not have the capability to directly operate member agent tools.",
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
        agent_team.print_response(user_input, stream=True)


# 测试模式
if __name__ == "__main__":
    respond()