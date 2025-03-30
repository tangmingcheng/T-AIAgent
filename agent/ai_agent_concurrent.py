import os

from agno.models.ollama import Ollama
from fastapi import FastAPI
from typing import Dict
from pydantic import BaseModel
from agno.agent import Agent
from agno.knowledge.combined import CombinedKnowledgeBase
from agno.storage.postgres import PostgresStorage
from agno.vectordb.pgvector import PgVector
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.knowledge.website import WebsiteKnowledgeBase
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.document.reader.pdf_reader import PDFReader
from agno.embedder.ollama import OllamaEmbedder
from agno.vectordb.search import SearchType
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.gmail import GmailTools
from agno.tools.yfinance import YFinanceTools
from agent.team import Team
from config.config import TOKEN_PATH, CREDENTIALS_PATH
import uuid

# Proxy settings
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# FastAPI app
app = FastAPI()

# Database configuration
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Global storage for team instances per user session
session_storage: Dict[str, Agent] = {}


# Request model
class TaskRequest(BaseModel):
    task: str


# Initialize shared components once
def initialize_shared_components():
    embedder = OllamaEmbedder(id="nomic-embed-text", dimensions=768)
    model =Groq(id='qwen-qwq-32b',timeout=60)

    return model, embedder


model, embedder = initialize_shared_components()


# Create a new team instance for each user session
def create_team(session_id: str):
    storage = PostgresStorage(
        table_name="agent_test_sessions",
        db_url=db_url,
        mode="agent"
    )

    gmail_tools = GmailTools(
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

    # Agents
    web_agent = Agent(
        name="Web Agent",
        role="Search the web for information",
        model=model,
        tools=[DuckDuckGoTools()],
        instructions="Always include sources",
        show_tool_calls=True,
        tool_choice="auto",
        markdown=True,
        debug_mode=False,
    )

    # Team
    agent_team = Agent(
        model=model,
        tools=[gmail_tools, DuckDuckGoTools(),],

        instructions=[
            "You can use GmailTools to perform email operations",
            "You can read the latest emails, unread emails, search emails, send emails, create drafts, and reply to emails",
            "You can use DuckDuckGoTools to find the latest information and provide link results",
            "when you use DuckDuckGoTools,you should always include sources",
            "Use tables to display data"
        ],
        # session_id="af42045d-e836-4257-bca6-dddef6a52119",
        storage=storage,
        add_history_to_messages=True,
        num_history_responses=10,
        update_knowledge=True,
        search_knowledge=True,
        show_tool_calls=True,
        markdown=True,
        debug_mode=False,
        add_datetime_to_instructions=True,
    )

    session_storage[session_id] = agent_team
    return agent_team


# HTTP endpoint for task processing using run
@app.post("/process_task")
async def process_task(request: TaskRequest):
    session_id = str(uuid.uuid4())
    team = create_team(session_id)

    response = await team.arun(request.task)

    # 只提取纯文本，避免包含无法序列化的对象
    if hasattr(response, "response"):  # 如果 response 是一个对象
        result = response.response.content
    else:
        result = str(response)

    del session_storage[session_id]
    print(f"response: {result}")
    return {"response": result}



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)