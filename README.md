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

Batch ask (for many questions from file):

```powershell
python -m niah.cli ask-batch --mode rag_sql_embeddings --namespace bench_001 --questions-file .\data\questions.txt
```

## Haystack Preparation

Prepare 15 PG essays from local files (recommended for reproducibility):

```powershell
python .\tools\prepare_pg_essays.py --mode from_local --local-dir .\data\pg_essays_raw --out-dir .\data\pg_essays
```

Optional direct URL fetch:

```powershell
python .\tools\prepare_pg_essays.py --mode from_urls --out-dir .\data\pg_essays
```

Build large haystack + inject needles:

```powershell
python .\tools\build_haystack.py --essays-dir .\data\pg_essays --needles-file .\data\needles.txt --global-repeat 8
```

Then ingest into RAG:

```powershell
python -m niah.cli ingest --namespace niah_bench --doc-id haystack1 --file .\data\full_haystack_with_needles.txt
```

## Logging

Each `ask` appends a JSON record to `NIAH_LOG_JSONL_PATH` (default `logs/runs.jsonl`):
- mode, namespace, question
- prompt/output char size
- prompt/completion token estimates
- latency
- retrieval metadata (for RAG mode)

This is designed so you can run your own benchmark harness later and evaluate drift/recall from logs.
