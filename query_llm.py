import json
import re
import os
import requests
from prompt_manager import get_operation_prompt
from openai import OpenAI


DeepSeek_API_KEY = os.getenv('DeepSeek_API_KEY')
client = OpenAI(api_key=DeepSeek_API_KEY, base_url="https://api.deepseek.com/v1")

def query_deepseek(user_input, parsed_content, stream=True):
    """调用 deepseek-r1 让 AI 识别用户输入并找出目标（流式输出）"""
    prompt = get_operation_prompt(user_input, parsed_content)

    if stream:
        # 流式输出
        response = client.chat.completions.create(
            model="deepseek-reasoner",  # 可改为 "deepseek-reasoner" 使用 DeepSeek-R1
            messages=[
                {"role": "system", "content": "你是一个智能助手，帮助用户理解意图并处理任务。"},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        responses = []
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:  # 确保 content 不为空
                responses.append(content)
                print(content, end='', flush=True)  # 实时打印，确保立即刷新输出缓冲区
        processed_text = ''.join(responses).replace(' ', '')
    else:
        # 非流式输出
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个智能助手，帮助用户理解意图并处理任务。"},
                {"role": "user", "content": prompt}
            ],
            stream=False,
        )
        processed_text = response.choices[0].message.content.replace(' ', '')
        print(processed_text)

        # 提取 JSON 数据
    structured_data = extract_json(processed_text)
    return structured_data



def query_ollama(user_input, parsed_content):
    """调用 deepseek-r1 让 AI 识别用户输入并找出目标（流式输出）"""
    prompt = get_operation_prompt(user_input, parsed_content)

    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {"model": "deepseek-r1:14b", "prompt": prompt, "stream": True}

    # 发起请求并启用流式传输
    response = requests.post(url, headers=headers, json=payload, stream=True)

    responses = []

    if response.status_code == 200:
        # 逐行读取响应数据，实时打印并保存
        for line in response.iter_lines():
            if line:
                try:
                    # 解析每一行的响应
                    response_data = json.loads(line)
                    # 提取 'response' 字段并保存
                    response_field = response_data.get("response", "")
                    responses.append(response_field)

                    # 实时输出每一部分的内容
                    print(response_field, end='', flush=True)

                except json.JSONDecodeError:
                    print("JSON解析错误")
    else:
        print(f"请求失败，错误码: {response.status_code}")

    # 将所有响应的文本拼接成一个字符串
    processed_text = ''.join(responses)

    # 移除多余的空格、换行符等
    processed_text = processed_text.replace(' ', '')

    # 通过提取的文本返回目标的结构化数据
    structured_data = extract_json(processed_text)

    return structured_data


def extract_json(model_result):
    """
    从模型返回的文本中提取结构化的JSON数据。
    """
    # 使用正则表达式提取 JSON 格式的数据
    match = re.search(r'\{.*?}', model_result, re.DOTALL)
    if match:
        try:
            # 输出匹配到的部分
            print("Matched JSON string:", match.group(0))
            # 尝试将提取的字符串解析为 JSON 数据
            json_data = json.loads(match.group(0))
            return json_data
        except json.JSONDecodeError:
            print("Error parsing JSON.")
            return None
    return None