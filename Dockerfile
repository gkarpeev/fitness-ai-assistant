FROM python:3.13-slim

# Рабочая директория внутри контейнера
WORKDIR /workspace

# Копируем зависимости
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Копируем весь проект внутрь контейнера (сохраняем структуру)
COPY . .

# Экспонируем порт Streamlit
EXPOSE 8501

# Запуск Streamlit (путь к скрипту в папке app)
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
