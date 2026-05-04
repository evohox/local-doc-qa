import pytest
from src.loader import load_document, split_documents
from langchain.schema import Document


def test_load_txt_file(sample_txt_file):
    chunks = load_document(sample_txt_file)

    assert len(chunks) > 0
    assert all(hasattr(c, "page_content") for c in chunks)
    assert all(len(c.page_content) > 0 for c in chunks)


def test_load_unsupported_format(tmp_path):
    fake_file = tmp_path / "test.docx"
    fake_file.write_text("content")

    with pytest.raises(ValueError, match="Неподдерживаемый формат"):
        load_document(str(fake_file))


def test_chunk_size_respected(sample_txt_file):
    chunks = load_document(sample_txt_file)

    for chunk in chunks:
        assert len(chunk.page_content) <= 1200


def test_chunks_preserve_metadata(sample_txt_file):
    chunks = load_document(sample_txt_file)

    for chunk in chunks:
        assert "source" in chunk.metadata


def test_split_documents_basic():
    docs = [Document(page_content="A" * 2000, metadata={"source": "test"})]
    chunks = split_documents(docs)

    assert len(chunks) > 1


def test_split_documents_basic():
    # Текст с реальными разделителями — как настоящий документ
    long_text = " ".join(["слово"] * 500)  # 500 слов через пробел > 1000 символов
    docs = [Document(page_content=long_text, metadata={"source": "test"})]
    chunks = split_documents(docs)

    assert len(chunks) > 1
