import os
import requests
import smtplib
from email.message import EmailMessage

tools = [
    {
        'type': 'function',
        'function': {
            'name': 'google_search',
            'description': 'Search for relevant information on the Internet.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'Search query string.',
                    },
                },
                'required': ['query'],
            },

        },

    },
    {
        'type': 'function',
        'function': {
            'name': 'send_email',
            'description': 'Send an email to the designated address.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'to': {
                        'type': 'string',
                        'description': "The recipient's email address."
                    },
                    'subject': {
                        "type": 'string',
                        'description': 'Email subject'
                    },
                    'body': {
                        'type': 'string',
                        'description': "The content of the body of the email."
                    }
                },
                'required': ['to', 'subject', 'body'],
            },

        },


    },
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': "Get the real time of the current user's timezone.",
            'parameters': {
                'type': 'object',
                'properties': {
                    'format': {
                        'type': 'string',
                        'description': "The format of time.eg:'date-time' or 'date' or 'time'",
                    },
                },
                'required': ['format'],

            },

        },

    },
    {
        'type': 'function',
        'function': {
            'name': 'execute_ui',
            'description': 'Executes UI automation tasks by analyzing the screen and clicking on target elements based on user commands.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'user_input': {
                        'type': 'string',
                        'description': "The user's operation instruction, such as 'Click on the recycle bin icon'."
                    },
                    'image_path': {
                        'type': 'string',
                        'description': "Path to the screenshot file used for UI analysis. Default: 'D:\\WallPaper\\test.png'.",
                        'default': 'D:\\WallPaper\\test.png'
                    },
                },
                'required': ['user_input'],
            },

        },


    },
]



# Google Search API 配置（若需要使用Google搜索）
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')  # Google API Key
GOOGLE_CX = os.getenv('GOOGLE_CX')      # Google search engine ID
GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"

def google_search(query):
    # 设置请求参数
    params = {
        'key': GOOGLE_API_KEY,
        'cx':  GOOGLE_CX,
        'q': query,
        'num' : 3,  # 返回前3条结果
    }
    # 发送GET请求
    try:
        resp = requests.get(GOOGLE_SEARCH_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("items", [])

        if not results:
            return "（未找到相关搜索结果。）"

        summary_lines = []
        for item in results:
            title = item.get("title", "无标题")
            snippet = item.get("snippet", "无摘要")
            link = item.get("link", "#")
            summary_lines.append(f"- [{title}]({link}): {snippet}")

            print(f"【Google 搜索结果】:\n" + "".join(summary_lines))
        return f"【Google 搜索结果】:\n" + "\n".join(summary_lines)
    except requests.RequestException as e:
        return f"（搜索请求失败: {e}。）"


# 配置你的 SMTP 服务器信息
SMTP_SERVER = "smtp.gmail.com"  # 如果是QQ邮箱，请改为 smtp.qq.com
SMTP_PORT = 465
SMTP_USER = os.getenv('SMTP_USER')  # Gmail
SMTP_PASS = os.getenv('SMTP_PASS')     # googl app password

def send_email(to: str, subject: str, body: str) -> str:
    """通过SMTP发送邮件，返回发送结果字符串。"""
    print(f"✅ 准备发送邮件至 {to}")

    try:
        msg = EmailMessage()
        msg["From"] = SMTP_USER
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SMTP_USER, SMTP_PASS)  # 登录 Gmail SMTP
            server.send_message(msg)  # 发送邮件

        print(f"✅ 邮件已成功发送至 {to}")
        return f"✅ 邮件已成功发送至 {to}"
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")
        return f"❌ 邮件发送失败: {e}"


def get_current_time(format):
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def nop(reason: str) -> str:
    """无操作函数：直接返回传入的reason文本。"""
    return reason


