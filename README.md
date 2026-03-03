# COMPASS_NIAH

Needle-in-a-Haystack benchmark scaffold with two API-driven modes:

1. `full_context`: sends full context in one prompt (no RAG).
2. `rag_sql_embeddings`: retrieves chunks from Postgres + pgvector.

The benchmark assets are English-only:
- `data/needles.txt`
- `data/questions.txt`
- generated haystacks/cases built from Paul Graham essays + English needles

## 1) Quick Start

```powershell
cd C:\development\COMPASS_NIAH
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill `.env`:
- `NIAH_LLM_ENDPOINT_URL`, `NIAH_LLM_API_KEY`, `NIAH_LLM_MODEL`
- optional for RAG mode: `NIAH_PG_DSN`

Recommended runtime env overrides for reproducibility:

```powershell
$env:NIAH_MAX_CONTEXT_TOKENS="500000"
$env:NIAH_LLM_MAX_OUTPUT_TOKENS="512"
$env:NIAH_LLM_TIMEOUT_S="180"
$env:NIAH_LLM_TEMPERATURE="0.0"
$env:NIAH_LLM_MAX_RETRIES="2"
$env:NIAH_LLM_RETRY_BACKOFF_S="2.0"
```

## 2) Reproduce Position Benchmark (Full Context)

This reproduces the current clean workflow (needle positions: 5/25/50/75/95).

```powershell
cd C:\development\COMPASS_NIAH
$env:NIAH_LOG_JSONL_PATH="logs/runs_posbench_rep5.jsonl"
$env:NIAH_PROMPT_IO_LOG_JSONL_PATH="logs/prompt_io_posbench_rep5.jsonl"

python .\tools\build_haystack.py --essays-dir .\data\pg_essays --needles-file .\data\needles.txt --global-repeat 5
python .\tools\build_position_cases.py --base-file .\data\full_base_haystack.txt --needles-file .\data\needles.txt --out-dir .\data\cases_rep5 --positions 0.05,0.25,0.50,0.75,0.95

$cases = Get-Content .\data\cases_rep5\cases_manifest.json | ConvertFrom-Json
$questions = Get-Content .\data\questions.txt
foreach ($c in $cases) {
  $idx=[int]$c.needle_index
  $q=$questions[$idx-1]
  $rid=("rep5_n{0:d2}_p{1:d2}" -f $idx,[int]$c.position_pct)
  $meta=@{needle_index=$idx;position_pct=[int]$c.position_pct;case_file=$c.file}|ConvertTo-Json -Compress
  python -m niah.cli ask --mode full_context --namespace niah_rep5 --context-file $c.file --question $q --run-id $rid --temperature 0.0 --meta-json $meta
}

python -m niah.cli eval-posbench --runs-file .\logs\runs_posbench_rep5.jsonl --needles-file .\data\needles.txt --run-prefix rep5_
```

Repeat with `global-repeat 3` and `2` by changing:
- log paths (`runs_posbench_rep3.jsonl`, `runs_posbench_rep2.jsonl`)
- case folder (`cases_rep3`, `cases_rep2`)
- run prefix (`rep3_`, `rep2_`)

## 3) Reproduce 10k Context Benchmark

```powershell
cd C:\development\COMPASS_NIAH
$env:NIAH_LOG_JSONL_PATH="logs/runs_posbench_10k_r01.jsonl"
$env:NIAH_PROMPT_IO_LOG_JSONL_PATH="logs/prompt_io_posbench_10k_r01.jsonl"
$env:NIAH_MAX_CONTEXT_TOKENS="20000"

$text = Get-Content .\data\full_base_haystack.txt -Raw
$maxChars = [Math]::Min($text.Length, 40000)
$base10k = $text.Substring(0, $maxChars)
Set-Content .\data\base_10k.txt $base10k

python .\tools\build_position_cases.py --base-file .\data\base_10k.txt --needles-file .\data\needles.txt --out-dir .\data\cases_10k --positions 0.05,0.25,0.50,0.75,0.95

$cases = Get-Content .\data\cases_10k\cases_manifest.json | ConvertFrom-Json
$questions = Get-Content .\data\questions.txt
foreach ($c in $cases) {
  $idx=[int]$c.needle_index
  $q=$questions[$idx-1]
  $rid=("k10_n{0:d2}_p{1:d2}_r01" -f $idx,[int]$c.position_pct)
  $meta=@{needle_index=$idx;position_pct=[int]$c.position_pct;case_file=$c.file}|ConvertTo-Json -Compress
  python -m niah.cli ask --mode full_context --namespace niah_10k --context-file $c.file --question $q --run-id $rid --temperature 0.0 --meta-json $meta
}

python -m niah.cli eval-posbench --runs-file .\logs\runs_posbench_10k_r01.jsonl --needles-file .\data\needles.txt --run-prefix k10_
```

## 4) Evaluation Commands

- Probe single run-set (one run per needle):
  - `python -m niah.cli eval-probes --needles-file .\data\needles.txt --run-prefix probe_en_v1 --match-mode sentence`
- Evaluate position benchmark (many runs):
  - `python -m niah.cli eval-posbench --runs-file .\logs\runs_posbench_rep5.jsonl --needles-file .\data\needles.txt --run-prefix rep5_`
- Print row-level decisions:
  - `python -m niah.cli eval-posbench --runs-file .\logs\runs_posbench_rep5.jsonl --needles-file .\data\needles.txt --run-prefix rep5_ --print-rows`

## 5) Logs and Where to Find Results

- Main run log (JSONL):
  - `NIAH_LOG_JSONL_PATH` (default `logs/runs.jsonl`)
- Full prompt I/O log (JSONL):
  - `NIAH_PROMPT_IO_LOG_JSONL_PATH` (default `logs/prompt_io.jsonl`)
- Position benchmark logs used in reproduction:
  - `logs/runs_posbench_rep5.jsonl`
  - `logs/runs_posbench_rep3.jsonl`
  - `logs/runs_posbench_rep2.jsonl`
  - `logs/runs_posbench_10k_r01.jsonl`

`logs/` is ignored by git (`.gitignore`), so benchmark artifacts stay local unless explicitly exported.

## 6) Notes

- Prompt budget is enforced before every call (`NIAH_MAX_CONTEXT_TOKENS`).
- For full-context benchmarks, add size guards to avoid accidental small-context runs:
  - `--expect-prompt-min <tokens>`
  - `--expect-prompt-max <tokens>`
  - Example: `python -m niah.cli ask --mode full_context ... --expect-prompt-min 240000 --expect-prompt-max 320000`
- If output is truncated, increase `NIAH_LLM_MAX_OUTPUT_TOKENS`.
- Run metadata now includes:
  - `llm_raw_completion_text`
  - `llm_finish_reason`
  - `completion_recovered_from_context`
  This makes it explicit whether truncation came from provider output or post-processing.
- Transient provider failures (429/502/503/504 + network timeouts) are retried automatically based on `NIAH_LLM_MAX_RETRIES` and `NIAH_LLM_RETRY_BACKOFF_S`.
- For RAG mode setup:
  - `python -m niah.cli init-db`
  - `python -m niah.cli ingest ...`
