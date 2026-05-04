import os
import pytest
import tempfile


@pytest.fixture
def sample_txt_file():
    content = """Искусственный интеллект (ИИ) — это способность машин выполнять задачи,
которые обычно требуют человеческого интеллекта.

Машинное обучение — это подраздел ИИ, который позволяет системам
автоматически обучаться на основе данных.

Нейронные сети — это вычислительные системы, вдохновлённые
биологическими нейронными сетями человеческого мозга."""

    # На Windows нельзя читать NamedTemporaryFile пока он открыт
    # Поэтому закрываем файл перед yield
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    f.write(content)
    f.close()  # явно закрываем до передачи в тест

    yield f.name

    os.unlink(f.name)  # удаляем после теста
