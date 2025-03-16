import os
from groq import Groq
#from groq_mock import Groq
import json
from ai_agent_omni import execute_ui
from prompt_manager import get_agent_prompt
from tools.tools import tools, google_search, send_email, get_current_time

# Initialize Groq client (using your provided API key)
client = Groq(api_key=os.getenv('GROQ_API_KEY'),timeout=60)

# 生成格式化后的 JSON 工具列表
tools_json = json.dumps(tools, indent=2, ensure_ascii=False)

available_functions = {
    'google_search': google_search,
    'send_email': send_email,
    'get_current_time': get_current_time,
    'execute_ui': execute_ui
}


def query_groq(messages: list) -> str:
    """Call Groq API to get LLM response for the given conversation messages."""
    while True:
        try:
            response = client.chat.completions.create(
                model='qwen-qwq-32b',  # 使用 Groq 支持的模型
                messages=messages,
                temperature=0.6,
                tools=tools,
                tool_choice="auto"  # Let model decide when to use tools
            )
        except Exception as e:
            print(f"❌ 调用 Groq API 失败: {e}")
            return "系统错误，请稍后重试。"


        # 确保 response.choices 存在且不为空
        if not response.choices or not hasattr(response.choices[0], 'message'):
            print("❌ Groq API 响应格式错误: response.choices 为空或缺少 message")
            return "API 响应异常，请稍后重试。"

        # Extract the response and any tool call responses
        response_message = response.choices[0].message

        # 获取模型返回的工具调用
        tool_calls = getattr(response_message, 'tool_calls', None)

        output = ''

        if tool_calls:

            # There may be multiple tool calls in the response
            for tool in tool_calls:
                # Ensure the function is available, and then call it
                if function_to_call := available_functions.get(tool.function.name):
                    print('🔹Calling function:', tool.function.name)
                    # 解析 JSON 结构，确保 `arguments` 是标准字典格式
                    try:
                        # 确保 arguments 是标准 Python 字典
                        raw_arguments = tool.function.arguments  # 直接从 tool 读取参数

                        if isinstance(raw_arguments, str):
                            print(f"❌ raw_arguments str")
                            arguments = json.loads(raw_arguments)  # 确保是 dict
                        elif isinstance(raw_arguments, dict):
                            print(f"✅ raw_arguments: dict")
                            arguments = raw_arguments  # 如果已经是 dict，直接使用
                        else:
                            arguments = {}  # 兜底防止 NoneType
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON 解析失败: {e}")
                        continue  # 跳过当前工具调用

                    print('📥Arguments:', arguments)
                    # 处理无参数函数
                    if arguments:
                        output = function_to_call(**arguments)


                    print('✅Function output:', output)

                    # Add the LLM's response to the conversation
                    messages.append({
                        "role": "assistant",
                        "content": response_message.content,
                        "tool_calls": [
                            {
                                "id": tool.id,
                                "function": {
                                    "name": tool.function.name,
                                    "arguments": tool.function.arguments
                                },
                                "type": "function"
                            }
                        ]
                    })
                    # Add the tool response to the conversation
                    messages.append({'tool_call_id': tool.id, 'role': "tool", 'name': tool.function.name,
                                     'content': json.dumps(output) if isinstance(output, (dict, list)) else str(output)})

                else:
                    print('❌Function', tool.function.name, 'not found')

        # Only needed to chat with the model using the tool call results
        if not tool_calls:
            print('No tool calls returned from model')
            assistant_reply = response.choices[0].message.content if hasattr(response_message, 'content') else ""
            assistant_reply = str(assistant_reply)
            break

    return assistant_reply


def agent_respond(messages: list):
    """Continuous conversation mode"""
    while True:
        user_input = input("💬 请输入你的问题（输入 'exit' 退出）：")
        if user_input.lower() in ["exit", "退出"]:
            print("🔚 结束对话。")
            break

        messages.append({'role': 'user', 'content': user_input})
        assistant_reply = query_groq(messages)

        if assistant_reply:
            print(f'\n🤖 AI: {assistant_reply}\n')
            messages.append({"role": "assistant", "content": assistant_reply})
        else:
            print("⚠️ AI 没有返回有效的响应，请重试。")


if __name__ == "__main__":
    print("🤖 欢迎使用 T-AIAgent 智能助手！输入 'exit' 退出对话。")

    prompt = get_agent_prompt()

    # 维护整个对话历史
    messages = [{'role': 'system', 'content': prompt}]

    # 进入对话主循环
    agent_respond(messages)