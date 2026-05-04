# Local Document Q&A

RAG-система для вопросов по своим документам. Загружаешь PDF или TXT,
задаёшь вопросы на естественном языке, получаешь ответы со ссылками
на источник и номер страницы.

## Стек

- **LangChain** — RAG цепочка и работа с документами
- **ChromaDB** — векторная база данных
- **Ollama** — локальная языковая модель (qwen2.5)
- **Streamlit** — веб-интерфейс

## Требования

- Python 3.11+
- [Ollama](https://ollama.com) с установленной моделью `qwen2.5`

## Установка

```bash
git clone https://github.com/evohox/local-doc-qa.git
cd local-doc-qa
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Запуск

```bash
streamlit run src/app.py
```

Открой браузер на `http://localhost:8501`

## Как использовать

1. Загрузи PDF или TXT через боковую панель
2. Нажми «Загрузить и индексировать»
3. Задай вопрос в поле «Вопрос»
4. Получи ответ со ссылкой на источник