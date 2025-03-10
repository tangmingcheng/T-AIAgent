import streamlit as st
import os
from ai_agent_rag_agno import update_knowledge_base,query_rag

# 初始化 Streamlit 页面
st.set_page_config(page_title="T-AI Agent聊天助手", page_icon="🤖")
st.title("🤖 T-AI Agent聊天助手")

# 初始化 Session State
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "custom_knowledge_base" not in st.session_state:
    st.session_state["custom_knowledge_base"] = None

# 添加文件上传功能
uploaded_file = st.file_uploader("上传一个 PDF 作为知识库", type=["pdf"])

if uploaded_file:
    # 保存文件到临时目录
    temp_dir = "temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)

    pdf_path = os.path.join(temp_dir, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"📄 文件 {uploaded_file.name} 已上传！正在解析...")

    # 重新加载知识库
    with st.spinner("🔄 正在处理文档..."):
        new_knowledge_base = update_knowledge_base(pdf_path)

    st.session_state["custom_knowledge_base"] = new_knowledge_base
    st.success("✅ 知识库已更新，AI 现在可以回答与你的 PDF 相关的问题！")

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 处理用户输入
if user_input := st.chat_input("请输入你的问题..."):
    # 显示用户输入
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)


    # 让 `agent` 回答
    with st.spinner('AI 正在思考中...'):
        assistant_reply = query_rag(user_input)

    if assistant_reply:
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    else:
        st.warning("AI 未返回有效的回复，请再试一次。")
