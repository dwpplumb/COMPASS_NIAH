from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):  # type: ignore
        return False


def _env_str(key: str, default: str = "") -> str:
    return str(os.getenv(key, default) or "").strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env_str(key, str(default)))
    except Exception:
        return int(default)


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env_str(key, str(default)))
    except Exception:
        return float(default)


def _env_list(key: str, default: str = "") -> list[str]:
    raw = _env_str(key, default)
    if not raw:
        return []
    return [part.strip() for part in raw.split("||") if part.strip()]


@dataclass(frozen=True)
class AppConfig:
    llm_endpoint_url: str
    llm_api_key: str
    llm_model: str
    llm_timeout_s: float
    llm_temperature: float
    llm_max_output_tokens: int
    llm_stop_sequences: list[str]
    llm_max_retries: int
    llm_retry_backoff_s: float
    embeddings_provider: str
    embeddings_endpoint_url: str
    embeddings_api_key: str
    embeddings_model: str
    embeddings_dim: int
    embeddings_timeout_s: float
    pg_dsn: str
    compass_prompt_file: str
    system_prompt: str
    max_context_tokens: int
    rag_top_k: int
    rag_chunk_chars: int
    rag_chunk_overlap_chars: int
    log_jsonl_path: str
    prompt_io_log_jsonl_path: str
    prompt_io_log_enabled: bool


def load_config() -> AppConfig:
    load_dotenv(override=False)
    return AppConfig(
        llm_endpoint_url=_env_str("NIAH_LLM_ENDPOINT_URL", "https://api.x.ai/v1/chat/completions"),
        llm_api_key=_env_str("NIAH_LLM_API_KEY"),
        llm_model=_env_str("NIAH_LLM_MODEL"),
        llm_timeout_s=_env_float("NIAH_LLM_TIMEOUT_S", 120.0),
        llm_temperature=_env_float("NIAH_LLM_TEMPERATURE", 0.0),
        llm_max_output_tokens=_env_int("NIAH_LLM_MAX_OUTPUT_TOKENS", 128),
        llm_stop_sequences=_env_list("NIAH_LLM_STOP_SEQUENCES", ""),
        llm_max_retries=_env_int("NIAH_LLM_MAX_RETRIES", 2),
        llm_retry_backoff_s=_env_float("NIAH_LLM_RETRY_BACKOFF_S", 2.0),
        embeddings_provider=_env_str("NIAH_EMBEDDINGS_PROVIDER", "deterministic").lower(),
        embeddings_endpoint_url=_env_str("NIAH_EMBEDDINGS_ENDPOINT_URL"),
        embeddings_api_key=_env_str("NIAH_EMBEDDINGS_API_KEY"),
        embeddings_model=_env_str("NIAH_EMBEDDINGS_MODEL"),
        embeddings_dim=_env_int("NIAH_EMBEDDINGS_DIM", 768),
        embeddings_timeout_s=_env_float("NIAH_EMBEDDINGS_TIMEOUT_S", 60.0),
        pg_dsn=_env_str("NIAH_PG_DSN"),
        compass_prompt_file=_env_str("NIAH_COMPASS_PROMPT_FILE"),
        system_prompt=_env_str("NIAH_SYSTEM_PROMPT", "You are a precise assistant. Answer strictly from provided context."),
        max_context_tokens=_env_int("NIAH_MAX_CONTEXT_TOKENS", 1_000_000),
        rag_top_k=_env_int("NIAH_RAG_TOP_K", 8),
        rag_chunk_chars=_env_int("NIAH_RAG_CHUNK_CHARS", 1200),
        rag_chunk_overlap_chars=_env_int("NIAH_RAG_CHUNK_OVERLAP_CHARS", 200),
        log_jsonl_path=_env_str("NIAH_LOG_JSONL_PATH", "logs/runs.jsonl"),
        prompt_io_log_jsonl_path=_env_str("NIAH_PROMPT_IO_LOG_JSONL_PATH", "logs/prompt_io.jsonl"),
        prompt_io_log_enabled=_env_str("NIAH_PROMPT_IO_LOG_ENABLED", "1").lower() in {"1", "true", "yes", "on"},
    )


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
