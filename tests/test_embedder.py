import pytest
from unittest.mock import patch, MagicMock
from langchain.schema import Document


def test_get_embeddings_uses_correct_model():
    with patch("src.embedder.OllamaEmbeddings") as mock_embeddings:
        from src.embedder import get_embeddings

        get_embeddings()
        mock_embeddings.assert_called_once()


def test_index_documents_calls_clear_first(mocker):
    mock_clear = mocker.patch("src.embedder.clear_vectorstore")
    mock_chroma = mocker.patch("src.embedder.Chroma.from_documents")
    mock_embeddings = mocker.patch("src.embedder.get_embeddings")

    from src.embedder import index_documents

    chunks = [Document(page_content="test", metadata={"source": "test.txt"})]
    index_documents(chunks)

    mock_clear.assert_called_once()
    mock_chroma.assert_called_once()


def test_index_documents_passes_chunks(mocker):
    mocker.patch("src.embedder.clear_vectorstore")
    mock_chroma = mocker.patch("src.embedder.Chroma.from_documents")
    mocker.patch("src.embedder.get_embeddings")

    from src.embedder import index_documents

    chunks = [
        Document(page_content="chunk 1", metadata={"source": "test.txt"}),
        Document(page_content="chunk 2", metadata={"source": "test.txt"}),
    ]
    index_documents(chunks)

    call_kwargs = mock_chroma.call_args
    assert call_kwargs.kwargs["documents"] == chunks
