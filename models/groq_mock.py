import os
import httpx
import json
from typing import Any, Union, Mapping


class ToolFunction:
    """模拟 Groq SDK 的工具调用 `function` 结构"""

    def __init__(self, function: dict):
        self.name = function.get("name", "")
        self.arguments = function.get("arguments", "{}")


class ToolCall:
    """模拟 Groq SDK 的工具调用 `tool_calls` 结构"""

    def __init__(self, tool_call: dict):
        self.id = tool_call.get("id", "")
        self.type = tool_call.get("type", "function")
        self.function = ToolFunction(tool_call.get("function", {}))


class ChatMessage:
    """模拟 Groq SDK 的 message 结构"""

    def __init__(self, message: dict):
        self.role = message.get("role", "assistant")
        self.content = message.get("content", "")
        self.tool_calls = [ToolCall(tc) for tc in message.get("tool_calls", [])]


class ChatChoice:
    """模拟 Groq SDK 的 choices 结构"""

    def __init__(self, choice: dict):
        self.index = choice.get("index", 0)
        self.message = ChatMessage(choice.get("message", {}))
        self.logprobs = choice.get("logprobs", None)
        self.finish_reason = choice.get("finish_reason", "stop")


class ChatCompletionResponse:
    """模拟 Groq SDK 返回的完整 response"""

    def __init__(self, data: dict):
        self.id = data.get("id")
        self.object = data.get("object")
        self.created = data.get("created")
        self.model = data.get("model")
        self.choices = [ChatChoice(choice) for choice in data.get("choices", [])]
        self.usage = data.get("usage", {})
        self.system_fingerprint = data.get("system_fingerprint", "")
        self.x_groq = data.get("x_groq", {})


class ChatCompletions:
    """模拟 `client.chat.completions` 部分"""

    def __init__(self, client):
        self.client = client

    def create(
        self,
        model: str,
        messages: list,
        temperature: float = 0.6,
        top_p: float = 1.0,
        n: int = 1,
        stream: bool = False,
        stop: Union[str, list, None] = None,
        max_tokens: Union[int, None] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: Union[dict, None] = None,
        user: Union[str, None] = None,
        tools: Union[list, None] = None,
        tool_choice: str = "auto",
    ) -> ChatCompletionResponse:
        """模拟 chat.completions.create() 方法"""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.client.api_key}"
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stream": stream,
            "stop": stop,
            "max_tokens": max_tokens,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "user": user
        }

        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()  # 如果 HTTP 状态码不是 2xx，则触发异常
            data = response.json()
            return ChatCompletionResponse(data)  # ✅ 确保返回 Groq SDK 兼容的对象
        except httpx.TimeoutException:
            raise RuntimeError("❌ 请求超时")
        except httpx.RequestError as e:
            raise RuntimeError(f"❌ 请求失败: {str(e)}")


class Chat:
    """模拟 `client.chat` 部分"""

    def __init__(self, client):
        self.completions = ChatCompletions(client)


class Groq:
    """模拟 `groq.Groq` 客户端"""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://api.groq.com",
        timeout: Union[float, None] = 10,
        max_retries: int = 3,
        default_headers: Union[Mapping[str, str], None] = None,
        default_query: Union[Mapping[str, Any], None] = None,
        http_client: Union[httpx.Client, None] = None,
        _strict_response_validation: bool = False,
    ):
        """构造新的 Groq 客户端"""
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("❌ API Key 未设置")

        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_headers = default_headers or {}
        self.default_query = default_query or {}

        self.chat = Chat(self)

    def copy(
        self,
        api_key: str = None,
        base_url: str = None,
        timeout: Union[float, None] = None,
        max_retries: int = None,
        default_headers: Union[Mapping[str, str], None] = None,
        default_query: Union[Mapping[str, Any], None] = None,
    ) -> "Groq":
        """复制当前客户端，并允许修改部分参数"""
        return Groq(
            api_key=api_key or self.api_key,
            base_url=base_url or self.base_url,
            timeout=timeout if timeout is not None else self.timeout,
            max_retries=max_retries if max_retries is not None else self.max_retries,
            default_headers=default_headers or self.default_headers,
            default_query=default_query or self.default_query,
        )


# **兼容官方 SDK 的方式**
Client = Groq


# **测试**
if __name__ == "__main__":
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    messages = [{"role": "user", "content": "gta6 最新消息"}]

    response = client.chat.completions.create(
        model="qwen-qwq-32b",
        messages=messages,
        tools=[{"type": "function", "function": {"name": "google_search", "parameters": {"query": "string"}}}],
        tool_choice="auto"
    )


    print("✅ AI 回复:", response.choices[0].message.content)

    # **打印工具调用**
    if response.choices[0].message.tool_calls:
        print("✅ 工具调用:")
        for tool in response.choices[0].message.tool_calls:
            print(f"🔹 工具 ID: {tool.id}, 名称: {tool.function.name}, 参数: {tool.function.arguments}")
