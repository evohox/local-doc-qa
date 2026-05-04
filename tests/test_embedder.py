import pytest
from unittest.mock import patch, MagicMock
from langchain.schema import Document


def test_get_embeddings_uses_correct_model():
    with patch("src.embedder.OllamaEmbeddings") as mock_embeddings:
        from src.embedder import get_embeddings

        get_embeddings()
        mock_embeddings.assert_called_once()


def test_index_documents_adds_to_existing(mocker):
    mock_vectorstore = MagicMock()
    mocker.patch("src.embedder.get_vectorstore", return_value=mock_vectorstore)
    mocker.patch("src.embedder.get_embeddings")

    from src.embedder import index_documents

    chunks = [Document(page_content="test", metadata={"source": "test.txt"})]
    index_documents(chunks, "test.txt")

    mock_vectorstore.add_documents.assert_called_once_with(chunks)


def test_index_documents_does_not_clear(mocker):
    mock_vectorstore = MagicMock()
    mocker.patch("src.embedder.get_vectorstore", return_value=mock_vectorstore)
    mocker.patch("src.embedder.get_embeddings")

    from src.embedder import index_documents

    chunks = [Document(page_content="test", metadata={"source": "test.txt"})]
    index_documents(chunks, "test.txt")

    assert (
        not hasattr(mock_vectorstore, "delete_collection")
        or not mock_vectorstore.delete_collection.called
    )


def test_get_indexed_documents_returns_unique_sources(mocker):
    mock_vectorstore = MagicMock()
    mock_vectorstore.get.return_value = {
        "metadatas": [
            {"source": "data/uploads/doc1.pdf"},
            {"source": "data/uploads/doc1.pdf"},
            {"source": "data/uploads/doc2.txt"},
        ],
        "ids": ["1", "2", "3"],
    }
    mocker.patch("src.embedder.get_vectorstore", return_value=mock_vectorstore)

    from src.embedder import get_indexed_documents

    docs = get_indexed_documents()

    assert len(docs) == 2
    assert "doc1.pdf" in docs
    assert "doc2.txt" in docs


def test_delete_document_removes_correct_chunks(mocker):
    mock_vectorstore = MagicMock()
    mock_vectorstore.get.return_value = {
        "metadatas": [
            {"source": "data/uploads/doc1.pdf"},
            {"source": "data/uploads/doc2.txt"},
        ],
        "ids": ["id1", "id2"],
    }
    mocker.patch("src.embedder.get_vectorstore", return_value=mock_vectorstore)

    from src.embedder import delete_document

    delete_document("doc1.pdf")

    mock_vectorstore.delete.assert_called_once_with(["id1"])
