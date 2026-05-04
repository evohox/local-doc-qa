import pytest
from unittest.mock import MagicMock
from langchain.schema import Document


@pytest.fixture
def mock_llm():
    return MagicMock()


def test_ask_returns_correct_structure(mocker, mock_llm):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "result": "ИИ это имитация человеческого интеллекта",
        "source_documents": [],
    }
    mocker.patch("src.retriever.get_qa_chain", return_value=mock_chain)
    mocker.patch(
        "src.retriever.get_vectorstore"
    ).return_value.similarity_search_with_score.return_value = [
        (
            Document(
                page_content="ИИ это...", metadata={"source": "test.txt", "page": 0}
            ),
            0.2,
        )
    ]

    from src.retriever import ask

    result = ask("Что такое ИИ?", mock_llm)

    assert "answer" in result
    assert "sources" in result
    assert "confidence" in result
    assert "score" in result
    assert isinstance(result["answer"], str)
    assert isinstance(result["sources"], list)


def test_ask_deduplicates_sources(mocker, mock_llm):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {"result": "Ответ", "source_documents": []}
    mocker.patch("src.retriever.get_qa_chain", return_value=mock_chain)
    mocker.patch(
        "src.retriever.get_vectorstore"
    ).return_value.similarity_search_with_score.return_value = [
        (Document(page_content="chunk 1", metadata={"source": "doc.txt"}), 0.3),
        (Document(page_content="chunk 2", metadata={"source": "doc.txt"}), 0.3),
        (Document(page_content="chunk 3", metadata={"source": "doc.txt"}), 0.3),
    ]

    from src.retriever import ask

    result = ask("Вопрос?", mock_llm)

    assert len(result["sources"]) == 1


def test_ask_formats_page_numbers(mocker, mock_llm):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {"result": "Ответ", "source_documents": []}
    mocker.patch("src.retriever.get_qa_chain", return_value=mock_chain)
    mocker.patch(
        "src.retriever.get_vectorstore"
    ).return_value.similarity_search_with_score.return_value = [
        (
            Document(page_content="chunk", metadata={"source": "doc.pdf", "page": 4}),
            0.2,
        ),
    ]

    from src.retriever import ask

    result = ask("Вопрос?", mock_llm)

    assert "стр. 5" in result["sources"][0]


def test_confidence_high(mocker, mock_llm):
    from src.retriever import get_confidence

    assert get_confidence(0.1)["label"] == "Высокая"


def test_confidence_medium(mocker, mock_llm):
    from src.retriever import get_confidence

    assert get_confidence(0.4)["label"] == "Средняя"


def test_confidence_low(mocker, mock_llm):
    from src.retriever import get_confidence

    assert get_confidence(0.8)["label"] == "Низкая"
