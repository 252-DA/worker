# Document Worker

Standalone async workers for the `document-chunk` pipeline.

## Prerequisites

- `uv`
- `../packages-ai` checked out next to this repo
- `../ai-sdk` available next to this repo

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
uv run content-generation-worker
uv run answer-grading-worker
```

`answer-grading-worker` consumes the `answer_grading` BullMQ queue: core-api
emits `ANSWER_GRADING_REQUESTED` when a submission contains short-answer
questions, and the worker grades each answer against the question frozen at
submit time, using the judge model (`LLM_PROFILE__QUIZ_VERIFICATION`). It runs
apart from `content-generation-worker` so learners waiting for a grade never
queue behind quiz generation. Set `WORKER_TYPE=answer_grading`.

`content-generation-worker` consumes the `content_generation` BullMQ queue,
retrieves LO-grounded source chunks from MCP, calls the configured model
through `ai-runtime-sdk`, validates the response, and persists quiz drafts.

Relevant environment variables:

```bash
WORKER_TYPE=content_generation
LEARNING_CONTEXT_MCP_ENABLED=true
LEARNING_CONTEXT_MCP_URL=http://localhost:8001/mcp
LEARNING_CONTEXT_MCP_TIMEOUT_SECONDS=30
LEARNING_CONTEXT_MCP_FALLBACK_TO_LOCAL=true

LLM__PROVIDER=gemini
LLM__MODEL=gemini-2.0-flash
LLM__API_KEY=...
```

Set `LLM__PROVIDER=openai-compatible` plus `LLM__BASE_URL` to use OpenAI,
LiteLLM, DeepSeek, Qwen, or another compatible chat-completions endpoint.

DeepSeek is also available as a first-class provider: set
`LLM__PROVIDER=deepseek`, `LLM__MODEL=deepseek-chat`, and `LLM__API_KEY`.
`LLM__BASE_URL` is optional and defaults to `https://api.deepseek.com`.

## Test

```bash
uv run pytest
```

## Docker

The Docker image also uses `uv` for dependency resolution:

```bash
docker compose -f docker-compose.worker.yml up --build
```
