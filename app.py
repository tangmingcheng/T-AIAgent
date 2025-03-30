import os
import streamlit as st
import tempfile

from agent.ai_agent_task_team import get_team_components

# 初始化函数
def initialize_team():
    return get_team_components()


# 检查是否已初始化
# 初始化检查
if "agent_team" not in st.session_state or "knowledge_agent" not in st.session_state:
    agent_team, knowledge_agent,local_pdf_knowledge_base= initialize_team()
    st.session_state.agent_team = agent_team
    st.session_state.knowledge_agent = knowledge_agent
    st.session_state.local_pdf_knowledge_base = local_pdf_knowledge_base


# 使用持久化的 agent_team
agent_team = st.session_state.agent_team
knowledge_agent = st.session_state.knowledge_agent
local_pdf_knowledge_base = st.session_state.local_pdf_knowledge_base

# Streamlit UI

st.set_page_config(page_title="AI Team Chat", layout="wide")
st.title("🧠 AI Team 聊天系统")

# 会话状态初始化
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 左侧上传文件扩展知识库
with st.sidebar:
    st.header("📂 上传 PDF 添加至知识库")
    uploaded_files = st.file_uploader("拖入 PDF 文件", type=["pdf"], accept_multiple_files=True)

    # 初始化一个标志来跟踪是否已处理当前上传
    if "files_processed" not in st.session_state:
        st.session_state.files_processed = False
    if "last_uploaded_files" not in st.session_state:
        st.session_state.last_uploaded_files = None

    # 只有当上传文件发生变化且未处理时才执行
    if uploaded_files and (uploaded_files != st.session_state.last_uploaded_files or not st.session_state.files_processed):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            local_pdf_knowledge_base.path = tmp_path
            local_pdf_knowledge_base.load(recreate=False, upsert=True)
            knowledge_agent.knowledge.load(recreate=False, upsert=True)
            st.success(f"✅ {uploaded_file.name} 已加入知识库")
            os.remove(tmp_path)  # 清理临时文件

        # 更新状态，表示当前文件已处理
        st.session_state.files_processed = True
        st.session_state.last_uploaded_files = uploaded_files

# 主界面聊天输入
user_input = st.chat_input("请输入问题或任务...")
chat_container = st.container()

# 展示历史聊天记录
for role, content in st.session_state.chat_history:
    with chat_container.chat_message(role):
        st.markdown(content)

# 处理新消息
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with chat_container.chat_message("user"):
        st.markdown(user_input)

    with chat_container.chat_message("assistant"):
        full_response = ""
        response_placeholder = st.empty()
        try:
            response_stream = agent_team.run(user_input, stream=True)

            for chunk in response_stream:
                if chunk.content:
                    full_response += chunk.content
                    response_placeholder.markdown(full_response)

            st.session_state.chat_history.append(("assistant", full_response))

        except Exception as e:
            st.error(f"❌ 出错了：{str(e)}")
            import traceback

            st.text(traceback.format_exc())


