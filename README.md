# Document Worker

Standalone async workers for the `document-chunk` pipeline.

## Prerequisites

- `uv`
- `../packages-ai` checked out next to this repo

## Install

```bash
cd worker
uv sync
```

`uv` resolves `document-chunk` from the local sibling path `../packages-ai`, so worker and core stay in sync during development.

## Run

```bash
uv run document-worker
uv run enrichment-worker
uv run outbox-worker
```

## Test

```bash
uv run pytest
```

## Docker

The Docker image also uses `uv` for dependency resolution:

```bash
docker compose -f docker-compose.worker.yml up --build
```
