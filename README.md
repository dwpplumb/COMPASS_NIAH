# COMPASS_NIAH

Needle-in-a-Haystack benchmark scaffold with two API-driven modes:

1. `full_context`: sends full context in one prompt (no RAG).
2. `rag_sql_embeddings`: retrieves chunks from Postgres+pgvector via embeddings.

This project is intentionally minimal and benchmark-oriented:
- no story/lore layer
- no evaluation scoring yet
- structured JSONL run logs for later offline analysis

## Setup

```powershell
cd C:\development\COMPASS_NIAH
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill `.env`:
- `NIAH_LLM_*` for xAI chat API (OpenAI-compatible endpoint)
- `NIAH_PG_DSN` for RAG mode
- optional `NIAH_EMBEDDINGS_*` for real embeddings
  - default provider is `deterministic` to keep setup flexible

## Commands

Initialize pgvector schema:

```powershell
python -m niah.cli init-db
```

Ingest file into embedding store:

```powershell
python -m niah.cli ingest --namespace bench_001 --doc-id haystack_a --file .\data\haystack.txt
```

Ask in full-context mode:

```powershell
python -m niah.cli ask --mode full_context --namespace bench_001 --context-file .\data\haystack.txt --question "What is the needle?"
```

Ask in RAG mode:

```powershell
python -m niah.cli ask --mode rag_sql_embeddings --namespace bench_001 --question "What is the needle?"
```

Inspect retrieval:

```powershell
python -m niah.cli debug-retrieve --namespace bench_001 --question "What is the needle?"
```

## Logging

Each `ask` appends a JSON record to `NIAH_LOG_JSONL_PATH` (default `logs/runs.jsonl`):
- mode, namespace, question
- prompt/output char size
- prompt/completion token estimates
- latency
- retrieval metadata (for RAG mode)

This is designed so you can run your own benchmark harness later and evaluate drift/recall from logs.
