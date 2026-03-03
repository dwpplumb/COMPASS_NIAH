from __future__ import annotations

import json
import time
from dataclasses import dataclass

from niah.config import AppConfig
from niah.embeddings import embed_texts
from niah.llm_client import call_chat_completion
from niah.models import LLMResponse
from niah.prompt_io_logger import make_prompt_io_record, write_prompt_io_record
from niah.prompting import build_full_context_user_prompt, build_rag_user_prompt, build_system_prompt, load_compass_text
from niah.run_logger import RunLogRecord, now_ts, write_record
from niah.text_clean import clean_completion_text, recover_full_sentence_from_context
from niah.token_estimator import estimate_tokens


@dataclass(frozen=True)
class AskResult:
    answer: str
    run_id: str
    prompt_tokens_est: int
    completion_tokens_est: int
    latency_ms: int


def _extract_finish_reason(raw: dict) -> str:
    try:
        choices = raw.get("choices", [])
        if isinstance(choices, list) and choices:
            reason = choices[0].get("finish_reason", "")
            return str(reason or "")
    except Exception:
        return ""
    return ""


def _enforce_prompt_budget(*, cfg: AppConfig, prompt_text: str, mode: str) -> None:
    max_tokens = int(cfg.max_context_tokens)
    if max_tokens <= 0:
        return
    prompt_tokens_est = estimate_tokens(prompt_text)
    if prompt_tokens_est > max_tokens:
        raise RuntimeError(
            f"Prompt too large for mode={mode}: estimated_tokens={prompt_tokens_est} exceeds "
            f"NIAH_MAX_CONTEXT_TOKENS={max_tokens}. Reduce context, switch to rag_sql_embeddings, "
            f"or raise NIAH_MAX_CONTEXT_TOKENS."
        )


def _mk_record(
    *,
    mode: str,
    namespace: str,
    question: str,
    prompt_text: str,
    response: LLMResponse,
    latency_ms: int,
    metadata: dict,
) -> RunLogRecord:
    prompt_tokens_est = estimate_tokens(prompt_text)
    completion_tokens_est = estimate_tokens(response.text)
    return RunLogRecord(
        run_id=metadata["run_id"],
        ts=now_ts(),
        mode=mode,
        namespace=namespace,
        question=question,
        input_chars=len(prompt_text),
        output_chars=len(response.text),
        prompt_tokens_est=prompt_tokens_est,
        completion_tokens_est=completion_tokens_est,
        total_tokens_est=prompt_tokens_est + completion_tokens_est,
        latency_ms=int(latency_ms),
        completion_text=response.text,
        metadata=metadata,
    )


def ask_full_context(
    *,
    cfg: AppConfig,
    namespace: str,
    context_text: str,
    question: str,
    run_id: str,
    temperature: float | None = None,
    extra_metadata: dict | None = None,
) -> AskResult:
    compass_text = load_compass_text(cfg)
    system_text = build_system_prompt(cfg=cfg, compass_text=compass_text)
    user_text = build_full_context_user_prompt(context_text=context_text, question=question)
    prompt_text_full = f"SYSTEM:\n{system_text}\n\nUSER:\n{user_text}"
    _enforce_prompt_budget(cfg=cfg, prompt_text=prompt_text_full, mode="full_context")
    t0 = time.time()
    llm = call_chat_completion(cfg=cfg, system_text=system_text, user_text=user_text, temperature=temperature)
    latency_ms = int((time.time() - t0) * 1000)
    raw_completion_text = str(llm.text)
    finish_reason = _extract_finish_reason(llm.raw)
    cleaned = clean_completion_text(raw_completion_text)
    recovered = recover_full_sentence_from_context(answer_text=cleaned, context_text=context_text)
    llm = LLMResponse(text=recovered, raw=llm.raw)
    rec = _mk_record(
        mode="full_context",
        namespace=namespace,
        question=question,
        prompt_text=prompt_text_full,
        response=llm,
        latency_ms=latency_ms,
        metadata={
            "run_id": run_id,
            "temperature": float(cfg.llm_temperature if temperature is None else temperature),
            "llm_finish_reason": finish_reason,
            "llm_raw_completion_text": raw_completion_text,
            "completion_recovered_from_context": bool(recovered != cleaned),
            "retrieved": [],
            **(extra_metadata or {}),
        },
    )
    write_record(jsonl_path=cfg.log_jsonl_path, rec=rec)
    if bool(cfg.prompt_io_log_enabled):
        io_rec = make_prompt_io_record(
            run_id=run_id,
            mode="full_context",
            namespace=namespace,
            question=question,
            system_text=system_text,
            user_text=user_text,
            prompt_text=prompt_text_full,
            completion_text=llm.text,
            raw_completion_text=raw_completion_text,
            metadata=rec.metadata,
        )
        write_prompt_io_record(jsonl_path=cfg.prompt_io_log_jsonl_path, rec=io_rec)
    return AskResult(
        answer=llm.text,
        run_id=run_id,
        prompt_tokens_est=rec.prompt_tokens_est,
        completion_tokens_est=rec.completion_tokens_est,
        latency_ms=latency_ms,
    )


def ask_rag_sql_embeddings(
    *,
    cfg: AppConfig,
    namespace: str,
    question: str,
    run_id: str,
    temperature: float | None = None,
    extra_metadata: dict | None = None,
) -> AskResult:
    if not cfg.pg_dsn:
        raise RuntimeError("Missing NIAH_PG_DSN for rag_sql_embeddings mode.")
    from niah.rag_store_pg import search_chunks

    qemb = embed_texts(cfg=cfg, texts=[question])[0]
    hits = search_chunks(
        dsn=cfg.pg_dsn,
        dim=cfg.embeddings_dim,
        namespace=namespace,
        query_embedding=qemb,
        limit=cfg.rag_top_k,
    )
    retrieved_text = "\n\n".join(
        f"[doc={h.doc_id} chunk={h.chunk_id} distance={h.distance:.6f}] {h.text}" for h in hits
    )
    compass_text = load_compass_text(cfg)
    system_text = build_system_prompt(cfg=cfg, compass_text=compass_text)
    user_text = build_rag_user_prompt(retrieved_context=retrieved_text, question=question)
    prompt_text_full = f"SYSTEM:\n{system_text}\n\nUSER:\n{user_text}"
    _enforce_prompt_budget(cfg=cfg, prompt_text=prompt_text_full, mode="rag_sql_embeddings")
    t0 = time.time()
    llm = call_chat_completion(cfg=cfg, system_text=system_text, user_text=user_text, temperature=temperature)
    latency_ms = int((time.time() - t0) * 1000)
    raw_completion_text = str(llm.text)
    finish_reason = _extract_finish_reason(llm.raw)
    cleaned = clean_completion_text(raw_completion_text)
    recovered = recover_full_sentence_from_context(answer_text=cleaned, context_text=retrieved_text)
    llm = LLMResponse(text=recovered, raw=llm.raw)
    rec = _mk_record(
        mode="rag_sql_embeddings",
        namespace=namespace,
        question=question,
        prompt_text=prompt_text_full,
        response=llm,
        latency_ms=latency_ms,
        metadata={
            "run_id": run_id,
            "temperature": float(cfg.llm_temperature if temperature is None else temperature),
            "llm_finish_reason": finish_reason,
            "llm_raw_completion_text": raw_completion_text,
            "completion_recovered_from_context": bool(recovered != cleaned),
            "retrieved_count": len(hits),
            "retrieved": [
                {"doc_id": h.doc_id, "chunk_id": h.chunk_id, "distance": h.distance, "meta_json": h.meta_json}
                for h in hits
            ],
            **(extra_metadata or {}),
        },
    )
    write_record(jsonl_path=cfg.log_jsonl_path, rec=rec)
    if bool(cfg.prompt_io_log_enabled):
        io_rec = make_prompt_io_record(
            run_id=run_id,
            mode="rag_sql_embeddings",
            namespace=namespace,
            question=question,
            system_text=system_text,
            user_text=user_text,
            prompt_text=prompt_text_full,
            completion_text=llm.text,
            raw_completion_text=raw_completion_text,
            metadata=rec.metadata,
        )
        write_prompt_io_record(jsonl_path=cfg.prompt_io_log_jsonl_path, rec=io_rec)
    return AskResult(
        answer=llm.text,
        run_id=run_id,
        prompt_tokens_est=rec.prompt_tokens_est,
        completion_tokens_est=rec.completion_tokens_est,
        latency_ms=latency_ms,
    )


def format_hits_for_stdout(cfg: AppConfig, namespace: str, question: str) -> str:
    if not cfg.pg_dsn:
        raise RuntimeError("Missing NIAH_PG_DSN")
    from niah.rag_store_pg import search_chunks

    qemb = embed_texts(cfg=cfg, texts=[question])[0]
    hits = search_chunks(
        dsn=cfg.pg_dsn,
        dim=cfg.embeddings_dim,
        namespace=namespace,
        query_embedding=qemb,
        limit=cfg.rag_top_k,
    )
    return json.dumps(
        [
            {
                "doc_id": h.doc_id,
                "chunk_id": h.chunk_id,
                "distance": h.distance,
                "meta_json": h.meta_json,
                "text_preview": h.text[:180],
            }
            for h in hits
        ],
        ensure_ascii=False,
        indent=2,
    )
