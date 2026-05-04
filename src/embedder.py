import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from src.logger import get_logger

load_dotenv()

logger = get_logger("embedder")

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/vectorstore")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")


def get_embeddings():
    return OllamaEmbeddings(model=OLLAMA_MODEL)


def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=get_embeddings(),
    )


def clear_vectorstore():
    logger.info("Очищаем векторную базу")
    vectorstore = get_vectorstore()
    vectorstore.delete_collection()


def index_documents(chunks: list) -> Chroma:
    logger.info(f"Индексируем {len(chunks)} чанков в ChromaDB")
    clear_vectorstore()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_DB_PATH,
    )
    logger.info("Индексация завершена")
    return vectorstore
