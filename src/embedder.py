import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/vectorstore")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")


def get_embeddings():
    return OllamaEmbeddings(model=OLLAMA_MODEL)


def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=get_embeddings(),
    )


def index_documents(chunks: list) -> Chroma:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_DB_PATH,
    )
    return vectorstore
