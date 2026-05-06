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
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def get_llm():
    return Ollama(
        model=OLLAMA_MODEL,
        temperature=0.1,
        base_url=OLLAMA_HOST,
    )


st.set_page_config(page_title="Local Document Q&A", layout="wide")
st.title("Local Document Q&A")
st.caption("Загрузи документы и задавай вопросы по их содержимому")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
                logger.info(f"Документ удалён: {doc}")
                st.rerun()

    st.divider()

    if st.button("Очистить историю"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            if message["sources"]:
                with st.expander("Источники"):
                    for source in message["sources"]:
                        st.markdown(f"- `{source}`")
            confidence = message.get("confidence")
            if confidence:
                st.caption(f"Релевантность: {confidence['icon']} {confidence['label']}")

question = st.chat_input("Задай вопрос по документам...")

if question:
    if not get_indexed_documents():
        st.warning("База пуста — сначала загрузи документы")
    else:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Думаю..."):
                try:
                    result = ask(question, get_llm(), st.session_state.chat_history)

                    st.write(result["answer"])

                    if result["sources"]:
                        with st.expander("Источники"):
                            for source in result["sources"]:
                                st.markdown(f"- `{source}`")

                    confidence = result["confidence"]
                    st.caption(
                        f"Релевантность: {confidence['icon']} {confidence['label']}"
                    )

                    st.session_state.chat_history.append((question, result["answer"]))

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                            "confidence": confidence,
                        }
                    )

                except Exception as e:
                    logger.error(f"Ошибка: {e}")
                    st.error(f"Ошибка: {e}")
