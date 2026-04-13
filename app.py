import streamlit as st
from analyzer import build_graph
from utils import read_file
from memory import create_memory, retrieve_memories, get_all_memories
from llm import get_llm

graph = build_graph()
llm = get_llm()

st.set_page_config(layout="wide")
st.sidebar.title("⚖️ Legal AI Pro")
menu = st.sidebar.radio("Navigation", ["NDA Analyzer", "Chat Assistant", "Memory", "Insights"])

st.title("⚖️ Legal AI (A-MEM Powered)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# NDA Analyzer
if menu == "NDA Analyzer":
    st.header("📄 Upload NDA")
    file = st.file_uploader("Upload NDA", type=["txt", "pdf"])

    if file and st.button("Analyze NDA"):
        text = read_file(file)
        result = graph.invoke({"document_text": text})
        output = result["output"]

        create_memory(text, output, "NDA Analysis")

        st.write(output)

# Chat Assistant
elif menu == "Chat Assistant":
    st.header("💬 Legal Chat")

    file = st.file_uploader("Upload document", type=["txt", "pdf"])
    user_input = st.chat_input("Ask something...")

    if user_input or file:
        text = read_file(file) if file else ""
        past = retrieve_memories(user_input or "document")

        prompt = f"""
        MEMORY:
        {past}

        DOCUMENT:
        {text}

        QUESTION:
        {user_input}
        """

        res = llm.invoke(prompt)
        output = res.content

        st.write(output)
        create_memory(user_input or text, output, "chat")

# Memory
elif menu == "Memory":
    data = get_all_memories()
    st.write(data)

# Insights
elif menu == "Insights":
    data = get_all_memories()
    st.write(data)