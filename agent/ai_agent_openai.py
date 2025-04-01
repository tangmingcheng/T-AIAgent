import os

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools

print(os.getenv('OPENAI_API_KEY'))
agent = Agent(
    model=OpenAIChat(id="gpt-4o",api_key=os.getenv('OPENAI_API_KEY')),
    description="You are an enthusiastic news reporter with a flair for storytelling!",
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True
)
agent.print_response("Tell me about a breaking news story from New York.", stream=True)