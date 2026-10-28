FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
RUN pip install --no-cache-dir ".[app,api]"
EXPOSE 8000 8501

CMD ["uvicorn", "finguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
