import streamlit as st
from ai_agent_ollama_tools import query_llm  # 你的 AI 代理
from prompt_manager import get_agent_prompt

prompt = get_agent_prompt()

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = [{'role': 'system',
                                  "content": prompt}]

# Streamlit 页面设置
st.set_page_config(page_title='T-AIAgent Chat', layout='wide')
st.title("🤖 T-AIAgent - AI Chat Interface")

# 聊天框
st.markdown("### Chat History:")
for message in st.session_state.messages:
    if message['role'] == 'user':
        st.markdown(f"**🧑‍💻 You:** {message['content']}")
    elif message['role'] == 'assistant':
        st.markdown(f"**🤖 AI:** {message['content']}")

# 用户输入框
user_input = st.text_input('Enter your message:', key='input')

# 发送消息
if st.button('Send'):
    if user_input:
        # 记录用户消息
        st.session_state.messages.append({'role': 'user', 'content': user_input})

        # 调用 AI 代理
        response = query_llm(st.session_state.messages)

        # 记录 AI 回复
        st.session_state.messages.append({'role': 'assistant', 'content': response})

        # 刷新页面以显示新消息
        st.rerun()

