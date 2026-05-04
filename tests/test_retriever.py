import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = "Тестовый ответ"
    return llm


def test_ask_returns_correct_structure(mocker, mock_llm):
    mocker.patch("src.retriever.search_documents", return_value=[])
    mocker.patch("src.retriever.build_context", return_value="контекст")

    from src.retriever import ask

    result = ask("Что такое ИИ?", mock_llm)

    assert "answer" in result
    assert "sources" in result
    assert "confidence" in result
    assert "score" in result


def test_ask_with_chat_history(mocker, mock_llm):
    mocker.patch("src.retriever.search_documents", return_value=[])
    mocker.patch("src.retriever.build_context", return_value="контекст")

    from src.retriever import ask

    history = [("Привет", "Привет!"), ("Как дела?", "Хорошо")]
    result = ask("Вопрос?", mock_llm, chat_history=history)

    assert "answer" in result


def test_ask_deduplicates_sources(mocker, mock_llm):
    from langchain.schema import Document

    docs = [
        (Document(page_content="chunk 1", metadata={"source": "doc.txt"}), 0.3),
        (Document(page_content="chunk 2", metadata={"source": "doc.txt"}), 0.3),
    ]
    mocker.patch("src.retriever.search_documents", return_value=docs)
    mocker.patch("src.retriever.build_context", return_value="контекст")

    from src.retriever import ask

    result = ask("Вопрос?", mock_llm)

    assert len(result["sources"]) == 1


def test_ask_formats_page_numbers(mocker, mock_llm):
    from langchain.schema import Document

    docs = [
        (
            Document(page_content="chunk", metadata={"source": "doc.pdf", "page": 4}),
            0.2,
        ),
    ]
    mocker.patch("src.retriever.search_documents", return_value=docs)
    mocker.patch("src.retriever.build_context", return_value="контекст")

    from src.retriever import ask

    result = ask("Вопрос?", mock_llm)

    assert "стр. 5" in result["sources"][0]


def test_confidence_high():
    from src.retriever import get_confidence

    assert get_confidence(0.1)["label"] == "Высокая"


def test_confidence_medium():
    from src.retriever import get_confidence

    assert get_confidence(0.4)["label"] == "Средняя"


def test_confidence_low():
    from src.retriever import get_confidence

    assert get_confidence(0.8)["label"] == "Низкая"
