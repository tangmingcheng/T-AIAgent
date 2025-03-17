import os
import json
import re
import time
from agno.agent import Agent, RunResponse
from agno.document.reader.pdf_reader import PDFReader
from agno.embedder.ollama import OllamaEmbedder
from agno.models.groq import Groq
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.pandas import PandasTools
from agno.tools.website import WebsiteTools
from agno.tools.gmail import GmailTools

from agno.knowledge.combined import CombinedKnowledgeBase
from agno.vectordb.pgvector import PgVector
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.knowledge.website import WebsiteKnowledgeBase
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.search import SearchType

from config.config import TOKEN_PATH
from config.config import CREDENTIALS_PATH
from config.config import DB_PATH

# Proxy Server（Clash/V2Ray/Trojan port）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

#task ="send email to 18340825516@163.com,subject:test google body:test"
#task ="Check out the latest email in my gmail mailbox"
#task ="What's the market outlook and financial performance of AI semiconductor companies? Then send email to .com"
#task ="I want to check the latest news on GTA6 and then send the collected information by email to 18340825516@163.com"
#task ="Take a look at https://ollama.com/"
#task ="I want to check the latest news on GTA6 and then repeat our previous conversation: "
#task ="Let's review our chat history"
task = "I would like to know what the knowledge base generally contains"

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Create a storage backend using the Sqlite database
storage = SqliteAgentStorage(
    # store sessions in the ai.sessions table
    table_name="agent_sessions",
    # db_file: Sqlite database file
    db_file=DB_PATH,
)
# Create Groq model
model=Groq(id='qwen-qwq-32b',timeout=60)

# Create Ollama embedder
embedder = OllamaEmbedder(id="nomic-embed-text", dimensions=768)

# Create knowledge  base
url_pdf_knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    # Table name: ai.pdf_documents
    vector_db=PgVector(
        table_name="pdf_documents",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=embedder,
    ),
)

website_knowledge_base = WebsiteKnowledgeBase(
    urls=["https://docs.agno.com/introduction"],
    # Number of links to follow from the seed URLs
    max_links=10,
    # Table name: ai.website_documents
    vector_db=PgVector(
        table_name="website_documents",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=embedder,
    ),
)

local_pdf_knowledge_base = PDFKnowledgeBase(
    path="D:/Chrome/Downloads/ThaiRecipes.pdf",
    # Table name: ai.pdf_documents
    vector_db=PgVector(
        table_name="pdf_documents",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=embedder,
    ),
    reader=PDFReader(chunk=True),
)

knowledge_base = CombinedKnowledgeBase(
    sources=[
        url_pdf_knowledge_base,
        website_knowledge_base,
        local_pdf_knowledge_base,
    ],
    vector_db=PgVector(
        # Table name: ai.combined_documents
        table_name="combined_documents",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=embedder,
    ),
)
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
        search_emails=True
    )

# Create reasoning agent
reason_agent = Agent(
    name="Reasoning Agent",
    role="Analyzes the user request and generates a structured execution plan",
    model=model,
    instructions=[
        "Analyze the user's input and break it down into simple steps based on natural language logic.",
        "Do not modify the user's input, simply divide it into distinct tasks as they are presented in the original request.",
        "Each task should correspond to one action mentioned in the user's input, and each action should be a single step.",
        "Return the steps as a JSON list, where each object has two fields: 'index' as an integer starting from 1, and 'description' as a string that directly repeats the user's request for each task.",
        "Ensure the steps are logically ordered and easy to follow."
    ],
    add_datetime_to_instructions=True,
    #reasoning=True,
    structured_outputs=True,
    markdown=True,
)

# Create implement agent
implement_agent = Agent(
    model=model,
    tools=[gmail_tools,DuckDuckGoTools(),WebsiteTools(),
           YFinanceTools(stock_price=True,historical_prices=True,analyst_recommendations=True, company_info=True),
           PandasTools()
    ],

    instructions=[
        "You can use GmailTools to perform email operations",
        "You can read the latest emails, unread emails, search emails, send emails, create drafts, and reply to emails",
        "You can use DuckDuckGoTools to find the latest information and provide link results",
        "when you use DuckDuckGoTools,you should always include sources",
        "You can use YFinanceTools to obtain data (current price or historical price) for a specified stock",
        "You can use PandasTools to perform statistical calculations if a set of numbers is provided",
        "Use tables to display data"
    ],
    #session_id="af42045d-e836-4257-bca6-dddef6a52119",
    storage=storage,
    add_history_to_messages=True,
    num_history_responses=10,
    knowledge=knowledge_base,
    search_knowledge=True,
    show_tool_calls=True,
    markdown=True,
    debug_mode= True,
    add_datetime_to_instructions=True,
)

# Comment out after the knowledge base is loaded
if implement_agent.knowledge is not None:
    implement_agent.knowledge.load(recreate=True)


# Extract json from agent response
def extract_json_from_run_response(response: RunResponse):
    """
    从 RunResponse.content 中提取 JSON 任务步骤
    """
    if not isinstance(response.content, str):
        print(f"❌ RunResponse.content not str，type: {type(response.content)}")
        return None

    # **使用正则表达式查找 JSON 代码块**
    json_match = re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', response.content)

    if json_match:
        json_data = json_match.group(0).strip()
        try:
            parsed_data = json.loads(json_data)  # 解析 JSON
            return parsed_data
        except json.JSONDecodeError:
            print("❌ JSON 解析失败，请检查数据格式")
            return None
    else:
        print("❌ 未找到 JSON 代码块")
        return None



# **定义 TaskExecutor（执行模块）**
class TaskExecutor:
    def __init__(self, _agent_team, max_retries=3):
        self.agent_team = _agent_team
        self.max_retries = max_retries  # 最大重试次数

    def execute(self, task_list):
        """ 逐步执行 reason_agent 解析出的任务，支持重试机制 """
        try:
            if not isinstance(task_list, list):
                raise ValueError("Invalid task structure, expected a list.")

            for step in task_list:
                step_index = step.get("index")
                step_description = step.get("description")

                # 重试机制
                retry_count = 0
                while retry_count < self.max_retries:
                    try:
                        print(f"⏳ 执行步骤 {step_index}: {step_description}")
                        self.agent_team.print_response(step_description, stream=True)
                        print(f"✅ 步骤 {step_index} 完成\n")
                        break  # 步骤成功，跳出重试循环
                    except Exception as e:
                        retry_count += 1
                        print(f"❌ 步骤 {step_index} 执行失败, 尝试 {retry_count}/{self.max_retries} 次")
                        if retry_count == self.max_retries:
                            print(f"❌ 步骤 {step_index} 执行失败 {self.max_retries} 次，跳过该步骤")
                            break  # 超过最大重试次数，跳过当前步骤

                    # 适当延迟，防止 API 速率限制
                    time.sleep(20)

        except Exception as e:
            print(f"❌ 任务执行失败: {str(e)}")


# Test

# **获取 Reason Agent 解析的任务**
task_plan = reason_agent.run(task, stream=False)  # 获取 RunResponse 对象
# **解析 JSON**
task_list = extract_json_from_run_response(task_plan)

# **打印解析结果**
if task_list:
    print("✅ 解析成功，任务步骤如下：")
    for step in task_list:
        print(f"Step {step['index']}: {step['description']}")
else:
    print("❌ 解析失败，未找到有效的任务数据")


# **执行任务**
executor = TaskExecutor(implement_agent)
executor.execute(task_list)