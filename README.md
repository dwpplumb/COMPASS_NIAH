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

Print answers during batch run:

```powershell
python -m niah.cli ask-batch --mode full_context --namespace bench_001 --questions-file .\data\questions.txt --context-file .\data\haystack.txt --print-answers
```

Probe all needles and print model answer + metrics:

```powershell
python -m niah.cli probe-needles --mode full_context --namespace bench_001 --needles-file .\data\needles.txt --context-file .\data\haystack.txt --temperature 0.0
```

Probe all needles with your own natural-language questions (same count as needles):

```powershell
python -m niah.cli probe-needles --mode full_context --namespace bench_001 --needles-file .\data\needles.txt --questions-file .\data\questions.txt --context-file .\data\haystack.txt --temperature 0.0
```

Show logged needle runs with expected needle text + model output:

```powershell
python -m niah.cli verify-logs --only-probe --needles-file .\data\needles.txt --last-n 5
```

Optional deterministic controls:

```powershell
python -m niah.cli ask --mode full_context --namespace bench_001 --context-file .\data\haystack.txt --question "What is the needle?" --run-id test_middle_50pct --temperature 0.0 --meta-json "{\"position_pct\":50,\"needle_id\":1}"
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

Create systematic position cases (5/25/50/75/95 by default):

```powershell
python .\tools\build_position_cases.py --base-file .\data\full_base_haystack.txt --needles-file .\data\needles.txt --out-dir .\data\cases
```

Then ingest into RAG:

```powershell
python -m niah.cli ingest --namespace niah_bench --doc-id haystack1 --file .\data\full_haystack_with_needles.txt
```

## Logging

Each `ask` appends a JSON record to `NIAH_LOG_JSONL_PATH` (default `logs/runs.jsonl`):
- mode, namespace, question
- completion_text (model output)
- prompt/output char size
- prompt/completion token estimates
- latency
- retrieval metadata (for RAG mode)

Full prompt I/O logging (enabled by default) writes one JSON record per run to
`NIAH_PROMPT_IO_LOG_JSONL_PATH` (default `logs/prompt_io.jsonl`) with:
- system_text
- user_text
- prompt_text (full assembled prompt)
- completion_text
- metadata

Disable if needed:

```powershell
$env:NIAH_PROMPT_IO_LOG_ENABLED="0"
```

This is designed so you can run your own benchmark harness later and evaluate drift/recall from logs.
