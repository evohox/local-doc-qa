import os
import time
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.embedder import get_vectorstore
from src.logger import get_logger

load_dotenv()

logger = get_logger("retriever")


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


def get_confidence(score: float) -> dict:
    if score <= 0.3:
        return {"label": "Высокая", "color": "green", "icon": "✓"}
    elif score <= 0.6:
        return {"label": "Средняя", "color": "orange", "icon": "~"}
    else:
        return {"label": "Низкая", "color": "red", "icon": "✗"}


def get_qa_chain(llm) -> RetrievalQA:
    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": build_prompt()},
    )

    return chain


def ask(question: str, llm) -> dict:
    logger.info(f"Вопрос: {question}")
    start = time.time()

    vectorstore = get_vectorstore()
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=3)

    avg_score = (
        sum(score for _, score in docs_with_scores) / len(docs_with_scores)
        if docs_with_scores
        else 1.0
    )
    confidence = get_confidence(avg_score)
    logger.info(
        f"Средний score релевантности: {round(avg_score, 3)} → {confidence['label']}"
    )

    chain = get_qa_chain(llm)
    result = chain.invoke({"query": question})

    elapsed = round(time.time() - start, 2)
    logger.info(f"Ответ получен за {elapsed}с")

    sources = []
    for doc, _ in docs_with_scores:
        source = doc.metadata.get("source", "неизвестно")
        page = doc.metadata.get("page", "")
        label = f"{source}, стр. {page + 1}" if page != "" else source
        if label not in sources:
            sources.append(label)

    logger.info(f"Источников найдено: {len(sources)}")

    return {
        "answer": result["result"],
        "sources": sources,
        "confidence": confidence,
        "score": round(avg_score, 3),
    }
