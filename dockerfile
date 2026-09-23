FROM python:3.12-slim

WORKDIR /app

COPY requirements .
RUN pip install --no-cache-dir -r requirements

COPY main.py .
COPY static static

# Bake the summarization model into the image so it isn't downloaded on first request.
RUN python -c "from transformers import pipeline; pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
