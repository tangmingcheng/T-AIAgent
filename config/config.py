import os

# 获取 agent 目录的绝对路径
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# token.json path
TOKEN_PATH = os.path.join(CONFIG_DIR, "token.json")

# credentials.json path
CREDENTIALS_PATH = os.path.join(CONFIG_DIR, "credentials.json")

# SQLite path
DB_PATH = os.path.join(CONFIG_DIR, "agent_sessions.db")
