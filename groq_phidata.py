import os
from typing import Optional, List, Iterator, Dict, Any, Mapping, Union
from phi.llm.base import LLM
from phi.llm.message import Message
from phi.utils.log import logger
from phi.utils.timer import Timer
from phi.utils.tools import get_function_call_for_tool_call
from pydantic import Field

try:
    from groq import Groq
except ImportError:
    logger.error("`groq` not installed")
    raise


class GroqLLM(LLM):
    """Groq API 适配器，兼容 Ollama 的接口，并继承 LLM 以适配 Assistant"""

    name: str = "Groq"
    model: str = "qwen-qwq-32b"
    timeout: Optional[Any] = None
    format: Optional[str] = None
    options: Optional[Any] = None
    keep_alive: Optional[Union[float, str]] = None
    client_kwargs: Optional[Dict[str, Any]] = None
    groq_client: Optional[Groq] = Field(default=None, exclude=True)

    def __init__(self, model: str = "qwen-qwq-32b", **kwargs):
        super().__init__()  # 初始化 pydantic.BaseModel

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables.")

        self.groq_client = Groq(api_key=api_key)
        self.model = model

        self.timeout = kwargs.get("timeout", None)
        self.format = kwargs.get("format", None)
        self.options = kwargs.get("options", None)
        self.keep_alive = kwargs.get("keep_alive", None)

    def api_kwargs(self) -> Dict[str, Any]:
        """构造 API 调用参数"""
        kwargs: Dict[str, Any] = {}
        if self.format is not None:
            kwargs["format"] = self.format
        if self.options is not None:
            kwargs["options"] = self.options
        if self.keep_alive is not None:
            kwargs["keep_alive"] = self.keep_alive
        return kwargs

    def to_llm_message(self, message: Message) -> Dict[str, Any]:
        """转换消息格式"""
        return {"role": message.role, "content": message.content}

    def invoke(self, messages: List[Message]) -> Mapping[str, Any]:
        """同步调用 Groq API 获取响应"""
        response_timer = Timer()
        response_timer.start()

        try:
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[self.to_llm_message(m) for m in messages],
                temperature=0.6,
                **self.api_kwargs()
            )
            response_timer.stop()
            logger.debug(f"Time to generate response: {response_timer.elapsed:.4f}s")


            if not response.choices or not isinstance(response.choices, list):
                logger.error(f"❌ Groq API 返回无效响应: {response}")
                return {"role": "assistant", "content": "API 响应异常，请稍后重试。"}

            first_choice = response.choices[0]

            if hasattr(first_choice, "message") and hasattr(first_choice.message, "content"):
                return {"role": "assistant", "content": first_choice.message.content.strip()}
            elif hasattr(first_choice, "content"):
                return {"role": "assistant", "content": first_choice.content.strip()}
            else:
                logger.error(f"❌ Groq API 解析失败: {response}")
                return {"role": "assistant", "content": "API 响应异常，请稍后重试。"}

        except Exception as e:
            logger.error(f"❌ 调用 Groq API 失败: {e}")
            return {"role": "assistant", "content": "系统错误，请稍后重试。"}

    def invoke_stream(self, messages: List[Message]) -> Iterator[Mapping[str, Any]]:
        """流式调用 Groq API 获取响应"""
        try:
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[self.to_llm_message(m) for m in messages],
                stream=True,  # ✅ 确保使用流模式
                temperature=0.6,
                **self.api_kwargs()
            )

            # ✅ 存储完整的响应
            full_response = ""

            for chunk in response:
                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]

                    # ✅ 正确解析 `delta.content`
                    if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                        content_piece = choice.delta.content
                        full_response += content_piece  # 拼接完整的流式响应
                        yield {"role": "assistant", "content": content_piece}
                    else:
                        logger.error(f"❌ Groq Stream 解析失败: {chunk}")
                else:
                    logger.error(f"❌ Groq API Stream 返回无效数据: {chunk}")

            #print(f"✅ Groq API Full Stream Response: {full_response}")

        except Exception as e:
            logger.error(f"❌ Groq Stream 调用失败: {e}")
            print(f"🔥 Groq Stream 调用失败: {e}")
            yield {"role": "assistant", "content": "系统错误，请稍后重试。"}

    def response(self, messages: List[Message]) -> str:
        """调用 Groq API 生成 LLM 响应"""
        response = self.invoke(messages)
        return response.get("content", "系统错误，请稍后重试。")

    def response_stream(self, messages: List[Message]) -> Iterator[str]:
        """流式响应"""
        for response in self.invoke_stream(messages):
            yield response.get("content", "")

    def model_dump(self) -> Dict[str, Any]:
        """解决 pydantic 需要序列化 LLM 的问题"""
        return {"name": self.name, "model": self.model}
