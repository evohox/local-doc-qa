import os
import time
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from src.embedder import get_vectorstore
from src.logger import get_logger

load_dotenv()

logger = get_logger("retriever")


def get_confidence(score: float) -> dict:
    if score <= 0.3:
        return {"label": "Высокая", "color": "green", "icon": "✓"}
    elif score <= 0.7:
        return {"label": "Средняя", "color": "orange", "icon": "~"}
    else:
        return {"label": "Низкая", "color": "red", "icon": "✗"}


def search_documents(question: str) -> list:
    vectorstore = get_vectorstore()
    all_data = vectorstore.get()

    sources_in_db = set()
    for meta in all_data["metadatas"]:
        if meta and "source" in meta:
            sources_in_db.add(meta["source"])

    docs_with_scores = []
    for source in sources_in_db:
        results = vectorstore.similarity_search_with_score(
            question,
            k=2,
            filter={"source": source},
        )
        docs_with_scores.extend(results)

    docs_with_scores.sort(key=lambda x: x[1])
    return docs_with_scores[:4]


def build_context(docs_with_scores: list) -> str:
    parts = []
    for doc, score in docs_with_scores:
        source = os.path.basename(doc.metadata.get("source", ""))
        page = doc.metadata.get("page", "")
        label = f"{source}, стр. {page + 1}" if page != "" else source
        parts.append(f"[{label}]\n{doc.page_content}")
    return "\n\n".join(parts)


def ask(question: str, llm, chat_history: list = []) -> dict:
    logger.info(f"Вопрос: {question}")
    start = time.time()

    docs_with_scores = search_documents(question)

    for doc, score in docs_with_scores:
        logger.info(
            f"Score: {round(score, 3)} | Source: {os.path.basename(doc.metadata.get('source', ''))} | Text: {doc.page_content[:80]}"
        )

    avg_score = (
        sum(s for _, s in docs_with_scores) / len(docs_with_scores)
        if docs_with_scores
        else 1.0
    )
    confidence = get_confidence(avg_score)
    logger.info(f"Score релевантности: {round(avg_score, 3)} → {confidence['label']}")

    context = build_context(docs_with_scores)

    history_text = ""
    if chat_history:
        lines = []
        for human, ai in chat_history[-3:]:
            lines.append(f"Человек: {human}")
            lines.append(f"Ассистент: {ai}")
        history_text = "\n".join(lines) + "\n\n"

    prompt = f"""Ты — помощник для ответов на вопросы по документам.
Используй ТОЛЬКО информацию из контекста ниже.
Если ответа нет в контексте — честно скажи об этом.
Отвечай на том же языке, на котором задан вопрос.

{history_text}Контекст:
{context}

Вопрос: {question}

Ответ:"""

    answer = llm.invoke(prompt)

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
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "score": round(avg_score, 3),
    }
