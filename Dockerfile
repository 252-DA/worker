FROM ghcr.io/astral-sh/uv:0.6.14 AS uv
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LLM__PROVIDER=gemini \
    LLM__MODEL=gemini-3-flash-preview

RUN apt-get update && \
    apt-get install -y --no-install-recommends libmagic1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=uv /uv /uvx /bin/

# Copy packages-ai as editable dependency
COPY packages-ai/pyproject.toml packages-ai/uv.lock* packages-ai/README.md /app/packages-ai/
COPY packages-ai/src/ /app/packages-ai/src/

# Copy shared AI runtime SDK
COPY ai-sdk/pyproject.toml ai-sdk/README.md /app/ai-sdk/
COPY ai-sdk/src/ /app/ai-sdk/src/

# Copy worker
COPY worker/pyproject.toml worker/uv.lock /app/worker/
COPY worker/worker/ /app/worker/worker/

WORKDIR /app/worker

RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "--frozen", "python", "-m"]
CMD ["worker.runners.document_worker"]
