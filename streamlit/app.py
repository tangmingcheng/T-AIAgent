import streamlit as st
from agent.ai_agent_groq_tools import query_groq
from prompt_manager import get_agent_prompt

# 初始化Streamlit页面配置
st.set_page_config(page_title="T-AI Agent聊天助手", page_icon="🤖")
st.title("🤖 T-AI Agent聊天助手")

# 初始化Session State
if "messages" not in st.session_state:
    st.session_state["messages"] = [{'role': 'system', 'content': get_agent_prompt()}]

# 显示历史消息
for message in st.session_state.messages:
    if message['role'] == 'user':
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message['role'] == 'assistant':
        with st.chat_message("assistant"):
            st.markdown(message["content"])

# 用户输入处理
if user_input := st.chat_input("请输入你的问题..."):
    # 显示用户的输入
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 调用ai_agent_ollama_tools获取响应
    with st.spinner('AI正在思考中...'):
        assistant_reply = query_groq(st.session_state.messages)

    assistant_reply = assistant_reply.strip()
    if assistant_reply:
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    else:
        st.warning("AI未返回有效的回复，请再试一次。")
