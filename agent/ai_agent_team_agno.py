import os
import json
import re
import time
from agno.agent import Agent, RunResponse
from agno.models.groq import Groq
from agno.storage.postgres import PostgresStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.pandas import PandasTools
from agno.tools.website import WebsiteTools
from agno.tools.gmail import GmailTools

# 代理服务器（Clash/V2Ray/Trojan 端口）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

#task ="send email to 18340825516@163.com,subject:test google body:test"
#task ="Check out the latest emails in my gmail mailbox"
#task ="What's the market outlook and financial performance of AI semiconductor companies? Then send email to .com"
task ="I want to check the latest news on GTA6 and then send the collected information by email to .com"
#task ="Take a look at https://ollama.com/"
#task ="I want to check the latest news on GTA6 and then repeat our previous conversation: "
#task ="回顾一下我们之间的聊天记录"

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
storage = PostgresStorage(table_name="agent_sessions", db_url=db_url)

gmail_tools=GmailTools(
        credentials_path="credentials.json",
        token_path="token.json",
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

# Create Groq model
model=Groq(id='qwen-qwq-32b',timeout=60)

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

# Create team member agent
web_search_agent  = Agent(
    name="Web Search Agent",
    role="Responsible for finding information through search engines",
    model=model,
    tools=[DuckDuckGoTools()],
    instructions=["Use DuckDuckGoTools to find the latest information and provide link results","Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

web_scraper_agent = Agent(
    name="Web Scraper Agent",
    role="Responsible for extracting web page content",
    model=model,
    tools=[WebsiteTools()],
    instructions=["Use the read_url function of WebsiteTools to read the main content of the specified URL"]
)

data_analysis_agent  = Agent(
    name="Data Analysis Agent",
    role="Responsible for acquiring data and conducting analysis",
    model=model,
    tools=[YFinanceTools(stock_price=True,historical_prices=True,analyst_recommendations=True, company_info=True),PandasTools()],
    instructions=[
        "Use YFinanceTools to obtain data (current price or historical price) for a specified stock",
        "If a set of numbers is provided, use PandasTools to perform statistical calculations"
    ],
    show_tool_calls=True,
    markdown=True,
)

email_agent = Agent(
    name="Email Agent",
    role="Responsible for Gmail email management",
    model=model,
    tools=[gmail_tools],
    instructions=[
        "Use GmailTools to perform email operations",
        "You can read the latest emails, unread emails, search emails, send emails, create drafts, and reply to emails",
    ],
    show_tool_calls=True,
    markdown=True,
)

# Create team leader agent
agent_team = Agent(
    name="Coordinator Agent",
    role="Intelligently dispatch dedicated agents to complete complex tasks",
    team=[web_search_agent, web_scraper_agent, data_analysis_agent, email_agent],
    tools=[gmail_tools],
    model=model,
    instructions=[
        "Do not attempt to perform any actions or use any tools. Simply determine which agent is most suited for each task and delegate accordingly.",
        "As the coordinator agent, you do not have the capability to perform tasks directly or use any tools yourself.",
    ],
    storage=storage,
    add_history_to_messages=True,
    show_tool_calls=True,
    markdown=True,
    debug_mode= True,
    add_datetime_to_instructions=True,
)

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
executor = TaskExecutor(agent_team)
executor.execute(task_list)