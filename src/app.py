import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from src.loader import load_document
from src.embedder import index_documents, get_indexed_documents, delete_document
from src.retriever import ask
from src.logger import get_logger

load_dotenv()

logger = get_logger("app")

UPLOAD_PATH = os.getenv("UPLOAD_PATH", "./data/uploads")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")


def get_llm():
    return Ollama(model=OLLAMA_MODEL, temperature=0.1)


st.set_page_config(page_title="Local Document Q&A", layout="wide")
st.title("Local Document Q&A")
st.caption("Загрузи документы и задавай вопросы по их содержимому")

with st.sidebar:
    st.header("Загрузить документ")
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
                index_documents(chunks, uploaded_file.name)

            logger.info(f"Файл загружен: {uploaded_file.name}")
            st.success(f"Готово. Чанков: {len(chunks)}")
            st.rerun()

    st.divider()

    st.header("База знаний")
    docs = get_indexed_documents()

    if not docs:
        st.caption("База пуста — загрузи документы")
    else:
        for doc in docs:
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"📄 {doc}")
            if col2.button("✕", key=f"del_{doc}"):
                delete_document(doc)
                logger.info(f"Документ удалён из базы: {doc}")
                st.rerun()

question = st.text_input("Вопрос", placeholder="О чём эти документы?")

if st.button("Спросить", type="primary"):
    if not question.strip():
        st.warning("Введи вопрос")
    elif not get_indexed_documents():
        st.warning("База пуста — сначала загрузи документы")
    else:
        try:
            llm = get_llm()
            with st.spinner("Думаю..."):
                result = ask(question, llm)

            st.subheader("Ответ")
            st.write(result["answer"])

            if result["sources"]:
                st.subheader("Источники")
                for source in result["sources"]:
                    st.markdown(f"- `{source}`")

        except Exception as e:
            logger.error(f"Ошибка: {e}")
            st.error(f"Ошибка: {e}")
