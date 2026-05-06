FROM python:3.11-slim

WORKDIR /app

# Устанавливаем системные зависимости для сборки C++ пакетов
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости отдельно — это кешируется Docker'ом
# Если requirements.txt не менялся — этот слой не пересобирается
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код проекта
COPY src/ ./src/
COPY .env.example .env

# Создаём папки для данных
RUN mkdir -p data/uploads data/vectorstore logs

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]