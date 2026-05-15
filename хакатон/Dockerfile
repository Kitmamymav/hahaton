# Используем легковесный образ Python
FROM python:3.10-slim

WORKDIR /app

# Копируем список зависимостей (создайте requirements.txt)
COPY requirements.txt .

# Устанавливаем библиотеки
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY . .

# Открываем порт Streamlit
EXPOSE 8501

# Команда для запуска приложения
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
