from agno.agent import Agent
from agno.media import Image
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools

model = Groq(id="llama-3.2-90b-vision-preview", timeout=60,temperature=1,top_p=1,stop=None, )

agent = Agent(
    model=model,
    create_default_system_message=False,  # 关键设置！
    markdown=True,
)

agent.print_response(
    "请描述这张图的内容",
    images=[Image(url="https://upload.wikimedia.org/wikipedia/commons/0/0c/GoldenGateBridge-001.jpg")],
    stream=True,
)
