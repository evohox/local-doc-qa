# Local Document Q&A

RAG-система для вопросов по локальным документам. Загружаешь PDF или TXT,
задаёшь вопросы на естественном языке, получаешь ответы со ссылками на источник
и номер страницы. Всё работает локально — без внешних API и облаков.

## Возможности

- Загрузка PDF и TXT документов
- Персистентная база знаний из нескольких документов
- Чат с историей диалога
- Оценка релевантности каждого ответа
- Логирование всех запросов с временем ответа
- Запуск через Docker с поддержкой GPU

## Стек

- **LangChain** — RAG цепочка и работа с документами
- **ChromaDB** — векторная база данных
- **Ollama** — локальная языковая модель (gemma2:9b)
- **nomic-embed-text** — модель для эмбеддингов
- **Streamlit** — веб-интерфейс
- **Docker** — контейнеризация

## Требования

- Python 3.11+
- [Ollama](https://ollama.com) с установленными моделями

```bash
ollama pull gemma2:9b
ollama pull nomic-embed-text
```

## Локальный запуск

```bash
git clone https://github.com/evohox/local-doc-qa.git
cd local-doc-qa
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
streamlit run src/app.py
```

Открой браузер на `http://localhost:8501`

## Запуск через Docker

```bash
docker compose up
```

Скачай модели в контейнер (первый раз):

```bash
docker exec -it local-doc-qa-ollama-1 ollama pull gemma2:9b
docker exec -it local-doc-qa-ollama-1 ollama pull nomic-embed-text
```

Открой браузер на `http://localhost:8501`

## Как использовать

1. Загрузи один или несколько PDF/TXT через боковую панель
2. Нажми «Загрузить и индексировать»
3. Задай вопрос в поле внизу экрана
4. Получи ответ со ссылкой на источник и оценкой релевантности

## Тесты

```bash
python -m pytest tests/ -v
```

## Структура проекта

```
src/
├── app.py        # Streamlit UI
├── loader.py     # загрузка и нарезка документов
├── embedder.py   # эмбеддинги и ChromaDB
├── retriever.py  # RAG цепочка
└── logger.py     # логирование
```