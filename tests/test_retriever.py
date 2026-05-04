import pytest
from unittest.mock import MagicMock
from langchain.schema import Document


def test_ask_returns_correct_structure(mocker):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "result": "ИИ это имитация человеческого интеллекта",
        "source_documents": [
            Document(
                page_content="ИИ это...", metadata={"source": "test.txt", "page": 0}
            )
        ],
    }

    mocker.patch("src.retriever.get_qa_chain", return_value=mock_chain)

    from src.retriever import ask

    result = ask("Что такое ИИ?")

    assert "answer" in result
    assert "sources" in result
    assert isinstance(result["answer"], str)
    assert isinstance(result["sources"], list)


def test_ask_deduplicates_sources(mocker):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "result": "Ответ",
        "source_documents": [
            Document(page_content="chunk 1", metadata={"source": "doc.txt"}),
            Document(page_content="chunk 2", metadata={"source": "doc.txt"}),
            Document(page_content="chunk 3", metadata={"source": "doc.txt"}),
        ],
    }

    mocker.patch("src.retriever.get_qa_chain", return_value=mock_chain)

    from src.retriever import ask

    result = ask("Вопрос?")

    assert len(result["sources"]) == 1


def test_ask_formats_page_numbers(mocker):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "result": "Ответ",
        "source_documents": [
            Document(page_content="chunk", metadata={"source": "doc.pdf", "page": 4})
        ],
    }

    mocker.patch("src.retriever.get_qa_chain", return_value=mock_chain)

    from src.retriever import ask

    result = ask("Вопрос?")

    assert "стр. 5" in result["sources"][0]
