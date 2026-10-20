# Container for the Agentic Research Assistant API.
#
# Build:  docker build -t research-assistant .
# Run:    docker run -p 8000:8000 --env-file .env research-assistant
# Then open http://localhost:8000/docs

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first, in their own layer, so Docker can cache them
# and skip reinstalling when only your code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Chroma's embedding model at build time, not on the first request
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warmup'])"

# Copy the code, the metadata database, and the vector index
# (.dockerignore keeps out the PDFs, your .env, and other junk)
COPY . .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
