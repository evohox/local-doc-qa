from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.logger import get_logger

logger = get_logger("loader")


def load_document(file_path: str) -> list:
    path = Path(file_path)
    logger.info(f"Загружаем файл: {path.name} ({path.suffix})")

    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    elif path.suffix.lower() == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        logger.error(f"Неподдерживаемый формат: {path.suffix}")
        raise ValueError(f"Неподдерживаемый формат: {path.suffix}")

    documents = loader.load()
    logger.info(f"Загружено страниц/секций: {len(documents)}")

    chunks = split_documents(documents)
    logger.info(f"Нарезано чанков: {len(chunks)}")
    return chunks


def split_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_documents(documents)
