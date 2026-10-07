FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 MODEL_CACHE=/app/models
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY prompts ./prompts
COPY app.py .
EXPOSE 8000 8501
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
