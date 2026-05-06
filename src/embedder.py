import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from src.logger import get_logger

load_dotenv()

logger = get_logger("embedder")

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/vectorstore")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def get_embeddings():
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_HOST,
    )


def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def index_documents(chunks: list, source_name: str) -> Chroma:
    logger.info(f"Индексируем {len(chunks)} чанков из '{source_name}'")

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)

    logger.info(f"Индексация завершена. Документ добавлен: {source_name}")
    return vectorstore


def get_indexed_documents() -> list[str]:
    vectorstore = get_vectorstore()
    result = vectorstore.get()

    sources = set()
    for meta in result["metadatas"]:
        if meta and "source" in meta:
            sources.add(os.path.basename(meta["source"]))

    return sorted(sources)


def delete_document(source_name: str):
    logger.info(f"Удаляем документ: {source_name}")

    vectorstore = get_vectorstore()
    result = vectorstore.get()

    ids_to_delete = []
    for i, meta in enumerate(result["metadatas"]):
        if meta and os.path.basename(meta.get("source", "")) == source_name:
            ids_to_delete.append(result["ids"][i])

    if ids_to_delete:
        vectorstore.delete(ids_to_delete)
        logger.info(f"Удалено чанков: {len(ids_to_delete)}")
    else:
        logger.warning(f"Документ не найден в базе: {source_name}")
