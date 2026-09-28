# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.6.14 AS uv
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LLM__PROVIDER=gemini \
    LLM__MODEL=gemini-3-flash-preview

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libmagic1 \
        zlib1g-dev \
        # opencv-python (docling OCR) needs these at import time; without
        # them OCR fails with "libGL.so.1: cannot open shared object file".
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=uv /uv /uvx /bin/

# Copy packages-ai as editable dependency
COPY packages-ai/pyproject.toml packages-ai/uv.lock* packages-ai/README.md /app/packages-ai/
COPY packages-ai/src/ /app/packages-ai/src/

# Copy shared AI runtime SDK
COPY ai-sdk/pyproject.toml ai-sdk/README.md /app/ai-sdk/
COPY ai-sdk/src/ /app/ai-sdk/src/

# Copy service SDK (GrpcEmbedder's da_service_sdk dependency, transitive via
# document-chunk[remote-embedding] — tool.uv.sources resolves it to ../service-sdk)
COPY service-sdk/pyproject.toml service-sdk/uv.lock* service-sdk/README.md /app/service-sdk/
COPY service-sdk/src/ /app/service-sdk/src/

# Copy worker
COPY worker/pyproject.toml worker/uv.lock /app/worker/
COPY worker/worker/ /app/worker/worker/

WORKDIR /app/worker

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Pin the tokenizer snapshot used to enforce the BGE-m3 input budget offline.
RUN .venv/bin/python -c "from tokenizers import Tokenizer; Tokenizer.from_pretrained('BAAI/bge-m3', revision='5617a9f61b028005a4858fdac845db406aefb181').save('/app/bge-m3-tokenizer.json')"
ENV CHUNKER__TOKENIZER_PATH=/app/bge-m3-tokenizer.json

ENTRYPOINT ["uv", "run", "--frozen", "python", "-m"]
CMD ["worker.runners.document_worker"]
