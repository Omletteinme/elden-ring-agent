# Backend-only image for Hugging Face Spaces (Docker SDK). Rebuilds the
# vector store from the committed chunks at build time -- data/chroma/ and
# bm25.pkl are gitignored (regeneratable), so the index doesn't exist until
# this runs. GROQ_API_KEY is provided as an HF Space "secret" at runtime,
# not baked into the image.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/chunks/chunks.jsonl ./data/chunks/chunks.jsonl

WORKDIR /app/src
RUN python index.py

EXPOSE 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
