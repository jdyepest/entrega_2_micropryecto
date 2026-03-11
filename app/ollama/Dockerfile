FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn httpx python-dotenv

COPY app/ollama/app.py /app/app.py
COPY .env /app/.env

EXPOSE 11434

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "11434"]
