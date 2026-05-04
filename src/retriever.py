import os
import time
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.embedder import get_vectorstore
from src.logger import get_logger

load_dotenv()

logger = get_logger("retriever")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")


def get_llm():
    return Ollama(model=OLLAMA_MODEL, temperature=0.1)


def build_prompt() -> PromptTemplate:
    template = """Ты — помощник для ответов на вопросы по документам.
Используй ТОЛЬКО информацию из контекста ниже.
Если ответа нет в контексте — честно скажи об этом.
Отвечай на том же языке, на котором задан вопрос.

Контекст:
{context}

Вопрос: {question}

Ответ:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )


def get_qa_chain() -> RetrievalQA:
    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": build_prompt()},
    )

    return chain


def ask(question: str) -> dict:
    logger.info(f"Вопрос: {question}")
    start = time.time()

    chain = get_qa_chain()
    result = chain.invoke({"query": question})

    elapsed = round(time.time() - start, 2)
    logger.info(f"Ответ получен за {elapsed}с")

    sources = []
    for doc in result["source_documents"]:
        source = doc.metadata.get("source", "неизвестно")
        page = doc.metadata.get("page", "")
        label = f"{source}, стр. {page + 1}" if page != "" else source
        if label not in sources:
            sources.append(label)

    logger.info(f"Источников найдено: {len(sources)}")

    return {
        "answer": result["result"],
        "sources": sources,
    }
