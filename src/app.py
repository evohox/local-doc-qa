import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil
import streamlit as st
from dotenv import load_dotenv
from src.loader import load_document
from src.embedder import index_documents
from src.retriever import ask

load_dotenv()

UPLOAD_PATH = os.getenv("UPLOAD_PATH", "./data/uploads")

st.set_page_config(page_title="Local Document Q&A", layout="wide")
st.title("Local Document Q&A")
st.caption("Загрузи документ и задавай вопросы по его содержимому")

with st.sidebar:
    st.header("Документ")
    uploaded_file = st.file_uploader("PDF или TXT", type=["pdf", "txt"])

    if st.button("Загрузить и индексировать", type="primary"):
        if uploaded_file is None:
            st.warning("Сначала выбери файл")
        else:
            os.makedirs(UPLOAD_PATH, exist_ok=True)
            dest = os.path.join(UPLOAD_PATH, uploaded_file.name)

            with open(dest, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Индексирую..."):
                chunks = load_document(dest)
                index_documents(chunks)

            st.success(f"Готово. Чанков: {len(chunks)}")

question = st.text_input("Вопрос", placeholder="О чём этот документ?")

if st.button("Спросить", type="primary"):
    if not question.strip():
        st.warning("Введи вопрос")
    else:
        with st.spinner("Думаю..."):
            result = ask(question)

        st.subheader("Ответ")
        st.write(result["answer"])

        if result["sources"]:
            st.subheader("Источники")
            for source in result["sources"]:
                st.markdown(f"- `{source}`")
