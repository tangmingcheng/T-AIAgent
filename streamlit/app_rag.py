import streamlit as st
import os
from agent.ai_agent_agno import update_knowledge_base, query_agent, knowledge_base

# 初始化 Streamlit 页面
st.set_page_config(page_title="T-AI Agent聊天助手", page_icon="🤖")
st.title("🤖 T-AI Agent聊天助手")

# 初始化 Session State
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "knowledge_base" not in st.session_state:
    st.session_state["knowledge_base"] = knowledge_base

# 添加文件上传功能
uploaded_file = st.file_uploader("上传一个 PDF 作为知识库", type=["pdf"])
if uploaded_file:
    temp_dir = "temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)
    pdf_path = os.path.join(temp_dir, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"📄 文件 {uploaded_file.name} 已上传！正在解析...")
    with st.spinner("🔄 正在处理文档..."):
        new_knowledge_base = update_knowledge_base(pdf_path)
    st.session_state["knowledge_base"] = new_knowledge_base
    st.success("✅ 知识库已更新，AI 现在可以回答与你的 PDF 相关的问题！")

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 处理用户输入并流式显示
if user_input := st.chat_input("请输入你的问题..."):
    # 显示用户输入
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 获取当前知识库并调用代理
    current_knowledge_base = st.session_state.get("knowledge_base", None)
    with st.chat_message("assistant"):
        with st.spinner("AI 正在思考中..."):
            response_stream = query_agent(user_input, knowledge_base_override=current_knowledge_base)
            assistant_reply = st.write_stream(response_stream)  # 流式显示并捕获完整响应

    # 保存完整回复到历史记录
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})