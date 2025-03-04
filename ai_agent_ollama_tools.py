import ollama
import json
from ai_agent_omni import execute_ui
from prompt_manager import get_agent_prompt
from tools import tools, google_search, send_email, get_current_time

# 生成格式化后的 JSON 工具列表
tools_json = json.dumps(tools, indent=2, ensure_ascii=False)

available_functions = {
  'google_search': google_search,
  'send_email': send_email,
  'get_current_time': get_current_time,
  'execute_ui':execute_ui
}


def query_llm(messages: list) -> str:
    """调用 Ollama 本地API获取 LLM 对指定对话消息的回复。"""
    while True:
        response =ollama.chat(
            model='qwen2.5:14b',
            messages=messages,
            options={'temperature':0.6},
            tools=tools,
        )

        tool_calls = response.message.tool_calls  # 获取模型返回的工具调用

        output = ''

        if tool_calls:
            # There may be multiple tool calls in the response
            for tool in tool_calls:
                # Ensure the function is available, and then call it
                if function_to_call := available_functions.get(tool.function.name):
                    print('🔹Calling function:', tool.function.name)
                    print('📥Arguments:', tool.function.arguments)
                    output = function_to_call(**tool.function.arguments)
                    print('✅Function output:', output)

                else:
                    print('❌Function', tool.function.name, 'not found')

        # Only needed to chat with the model using the tool call results
        if response.message.tool_calls:
            # Add the function response to messages for the model to use
            messages.append(response.message)
            messages.append({'role': 'tool', 'content': str(output), 'name': tool.function.name})

            # Get final response from model with function outputs
            #final_response = chat('qwen2.5:14b', messages=messages,tools=tools)
            #print('Final response🚀:', final_response.message.content)
            #assistant_reply=final_response.message.content
        else:
            print('No tool calls returned from model')
            assistant_reply=response.message.content
            #print('Final response🚀:', assistant_reply)
            break

    return assistant_reply


def agent_respond(messages: list):
    """持续对话模式"""
    while True:
        user_input = input("💬 请输入你的问题（输入 'exit' 退出）：")
        if user_input.lower() in ["exit", "退出"]:
            print("🔚 结束对话。")
            break

        messages.append({'role': 'user', 'content': user_input})
        assistant_reply = query_llm(messages)

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