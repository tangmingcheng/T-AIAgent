import os
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.tools.gmail import GmailTools
from agno.tools.yfinance import YFinanceTools

from agent.team import Team
from config.config import TOKEN_PATH, CREDENTIALS_PATH, DB_PATH

# Proxy Server（Clash/V2Ray/Trojan port）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

task ="Check the latest news on GTA6 and then send the collected information by email to 18340825516@163.com"

# Create a storage backend using the Sqlite database
storage = SqliteAgentStorage(
    table_name="agent_sessions",
    # db_file: Sqlite database file
    db_file=DB_PATH,
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

agent_team = Team(
    members=[web_agent, finance_agent,email_agent],
    name="Content Team",
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

agent_team.print_response(task, stream=True)
