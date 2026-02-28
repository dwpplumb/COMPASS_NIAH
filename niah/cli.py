from __future__ import annotations

import argparse
import json
from pathlib import Path

from niah.chunking import chunk_text
from niah.config import load_config
from niah.embeddings import embed_texts
from niah.engine import ask_full_context, ask_rag_sql_embeddings, format_hits_for_stdout
from niah.run_logger import new_run_id


def _read_text(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8")


def cmd_init_db() -> int:
    from niah.rag_store_pg import init_schema

    cfg = load_config()
    if not cfg.pg_dsn:
        raise RuntimeError("Missing NIAH_PG_DSN")
    init_schema(dsn=cfg.pg_dsn, dim=cfg.embeddings_dim)
    print(json.dumps({"ok": True, "dim": cfg.embeddings_dim}))
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from niah.rag_store_pg import upsert_chunks

    cfg = load_config()
    if not cfg.pg_dsn:
        raise RuntimeError("Missing NIAH_PG_DSN")
    text = _read_text(args.file)
    chunks = chunk_text(text=text, chunk_chars=cfg.rag_chunk_chars, overlap_chars=cfg.rag_chunk_overlap_chars)
    embeddings = embed_texts(cfg=cfg, texts=chunks)
    rows = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings), start=1):
        chunk_id = f"c_{i:05d}"
        rows.append((chunk_id, chunk, emb, {"source_file": str(args.file)}))
    written = upsert_chunks(
        dsn=cfg.pg_dsn,
        dim=cfg.embeddings_dim,
        namespace=str(args.namespace),
        doc_id=str(args.doc_id),
        rows=rows,
    )
    print(
        json.dumps(
            {"ok": True, "namespace": args.namespace, "doc_id": args.doc_id, "chunks": len(chunks), "rows_upserted": written},
            ensure_ascii=False,
        )
    )
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    cfg = load_config()
    run_id = str(args.run_id).strip() if str(args.run_id or "").strip() else new_run_id()
    extra_meta = {}
    if str(args.meta_json or "").strip():
        extra_meta = json.loads(str(args.meta_json))
        if not isinstance(extra_meta, dict):
            raise ValueError("--meta-json must be a JSON object")
    if args.mode == "full_context":
        if not args.context_file:
            raise RuntimeError("--context-file is required for mode=full_context")
        context_text = _read_text(args.context_file)
        out = ask_full_context(
            cfg=cfg,
            namespace=args.namespace,
            context_text=context_text,
            question=args.question,
            run_id=run_id,
            temperature=args.temperature,
            extra_metadata=extra_meta,
        )
    elif args.mode == "rag_sql_embeddings":
        out = ask_rag_sql_embeddings(
            cfg=cfg,
            namespace=args.namespace,
            question=args.question,
            run_id=run_id,
            temperature=args.temperature,
            extra_metadata=extra_meta,
        )
    else:
        raise RuntimeError(f"Unsupported mode: {args.mode}")

    print(out.answer)
    print(
        json.dumps(
            {
                "run_id": out.run_id,
                "prompt_tokens_est": out.prompt_tokens_est,
                "completion_tokens_est": out.completion_tokens_est,
                "total_tokens_est": out.prompt_tokens_est + out.completion_tokens_est,
                "latency_ms": out.latency_ms,
                "log_jsonl_path": cfg.log_jsonl_path,
                "temperature": args.temperature if args.temperature is not None else cfg.llm_temperature,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _load_questions(path: str) -> list[str]:
    p = Path(path)
    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, list):
            raise ValueError("questions JSON must be a list of strings")
        return [str(x).strip() for x in obj if str(x).strip()]
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _load_needles(path: str) -> list[str]:
    p = Path(path)
    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, list):
            raise ValueError("needles JSON must be a list of strings")
        return [str(x).strip() for x in obj if str(x).strip()]
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _read_jsonl(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"JSONL file not found: {path}")
    rows: list[dict] = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        line = ln.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def cmd_ask_batch(args: argparse.Namespace) -> int:
    cfg = load_config()
    questions = _load_questions(args.questions_file)
    if not questions:
        raise RuntimeError("No questions found.")

    for q in questions:
        run_id = new_run_id()
        extra_meta = {"batch_file": str(args.questions_file)}
        if args.mode == "full_context":
            if not args.context_file:
                raise RuntimeError("--context-file is required for mode=full_context")
            context_text = _read_text(args.context_file)
            out = ask_full_context(
                cfg=cfg,
                namespace=args.namespace,
                context_text=context_text,
                question=q,
                run_id=run_id,
                temperature=args.temperature,
                extra_metadata=extra_meta,
            )
        else:
            out = ask_rag_sql_embeddings(
                cfg=cfg,
                namespace=args.namespace,
                question=q,
                run_id=run_id,
                temperature=args.temperature,
                extra_metadata=extra_meta,
            )
        print(
            json.dumps(
                {
                    "run_id": out.run_id,
                    "mode": args.mode,
                    "namespace": args.namespace,
                    "question": q,
                    "prompt_tokens_est": out.prompt_tokens_est,
                    "completion_tokens_est": out.completion_tokens_est,
                    "total_tokens_est": out.prompt_tokens_est + out.completion_tokens_est,
                    "latency_ms": out.latency_ms,
                    "temperature": args.temperature if args.temperature is not None else cfg.llm_temperature,
                },
                ensure_ascii=False,
            )
        )
        if bool(args.print_answers):
            print(f"ANSWER[{out.run_id}]: {out.answer}")
    return 0


def cmd_probe_needles(args: argparse.Namespace) -> int:
    cfg = load_config()
    needles = _load_needles(args.needles_file)
    if not needles:
        raise RuntimeError("No needles found.")
    questions: list[str] = []
    if str(args.questions_file or "").strip():
        questions = _load_questions(str(args.questions_file))
        if len(questions) != len(needles):
            raise RuntimeError(
                f"questions count ({len(questions)}) must match needles count ({len(needles)}) for probe-needles."
            )

    context_text = ""
    if args.mode == "full_context":
        if not args.context_file:
            raise RuntimeError("--context-file is required for mode=full_context")
        context_text = _read_text(args.context_file)

    total = 0
    for i, _needle in enumerate(needles, start=1):
        total += 1
        run_id = f"{args.run_prefix}_{i:03d}" if str(args.run_prefix).strip() else new_run_id()
        q = questions[i - 1] if questions else (
            f"What is the exact content of NIAH_NEEDLE_{i}? "
            'If you cannot find it, answer exactly "UNSURE".'
        )
        meta = {"probe_type": "needle_probe", "needle_index": i, "question": q}
        if args.mode == "full_context":
            out = ask_full_context(
                cfg=cfg,
                namespace=args.namespace,
                context_text=context_text,
                question=q,
                run_id=run_id,
                temperature=args.temperature,
                extra_metadata=meta,
            )
        else:
            out = ask_rag_sql_embeddings(
                cfg=cfg,
                namespace=args.namespace,
                question=q,
                run_id=run_id,
                temperature=args.temperature,
                extra_metadata=meta,
            )
        print(f"ANSWER[{run_id}][needle={i}]: {out.answer}")
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "mode": args.mode,
                    "needle_index": i,
                    "latency_ms": out.latency_ms,
                    "total_tokens_est": out.prompt_tokens_est + out.completion_tokens_est,
                },
                ensure_ascii=False,
            )
        )

    print(
        json.dumps(
            {
                "ok": True,
                "mode": args.mode,
                "needles_total": total,
            },
            ensure_ascii=False,
        )
    )
    return 0


def cmd_debug_retrieve(args: argparse.Namespace) -> int:
    cfg = load_config()
    print(format_hits_for_stdout(cfg=cfg, namespace=args.namespace, question=args.question))
    return 0


def cmd_verify_logs(args: argparse.Namespace) -> int:
    cfg = load_config()
    runs_file = str(args.runs_file).strip() if str(args.runs_file).strip() else cfg.log_jsonl_path
    rows = _read_jsonl(runs_file)
    needles: list[str] = []
    if str(args.needles_file or "").strip():
        needles = _load_needles(str(args.needles_file))

    filtered = []
    for row in rows:
        meta = row.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {}
        if bool(args.only_probe) and str(meta.get("probe_type", "")) != "needle_probe":
            continue
        filtered.append(row)

    if not filtered:
        print(json.dumps({"ok": True, "runs_file": runs_file, "count": 0}, ensure_ascii=False))
        return 0

    if int(args.last_n or 0) > 0:
        filtered = filtered[-int(args.last_n):]

    for row in filtered:
        meta = row.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {}
        idx_raw = meta.get("needle_index", 0)
        try:
            idx = int(idx_raw)
        except Exception:
            idx = 0
        expected = needles[idx - 1] if idx > 0 and idx <= len(needles) else ""
        out = {
            "run_id": row.get("run_id", ""),
            "mode": row.get("mode", ""),
            "needle_index": idx,
            "needle": expected,
            "question": row.get("question", ""),
            "completion_text": row.get("completion_text", ""),
            "latency_ms": row.get("latency_ms", 0),
            "total_tokens_est": row.get("total_tokens_est", 0),
        }
        print(json.dumps(out, ensure_ascii=True))

    print(json.dumps({"ok": True, "runs_file": runs_file, "count": len(filtered)}, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="COMPASS_NIAH benchmark CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("init-db", help="Initialize Postgres + pgvector schema")
    p0.set_defaults(fn=lambda a: cmd_init_db())

    p1 = sub.add_parser("ingest", help="Ingest a text file into RAG store")
    p1.add_argument("--namespace", required=True)
    p1.add_argument("--doc-id", required=True)
    p1.add_argument("--file", required=True)
    p1.set_defaults(fn=cmd_ingest)

    p2 = sub.add_parser("ask", help="Ask question in chosen mode")
    p2.add_argument("--mode", required=True, choices=["full_context", "rag_sql_embeddings"])
    p2.add_argument("--namespace", required=True)
    p2.add_argument("--question", required=True)
    p2.add_argument("--context-file", default="", help="Required for full_context mode")
    p2.add_argument("--run-id", default="", help="Optional fixed run id")
    p2.add_argument("--temperature", type=float, default=None, help="Optional override")
    p2.add_argument("--meta-json", default="", help="Optional JSON object stored in log metadata")
    p2.set_defaults(fn=cmd_ask)

    p2b = sub.add_parser("ask-batch", help="Run multiple questions from file in chosen mode")
    p2b.add_argument("--mode", required=True, choices=["full_context", "rag_sql_embeddings"])
    p2b.add_argument("--namespace", required=True)
    p2b.add_argument("--questions-file", required=True, help="TXT (one question per line) or JSON list")
    p2b.add_argument("--context-file", default="", help="Required for full_context mode")
    p2b.add_argument("--temperature", type=float, default=None, help="Optional override")
    p2b.add_argument("--print-answers", action="store_true", help="Print model answers to stdout after each JSON row")
    p2b.set_defaults(fn=cmd_ask_batch)

    p2c = sub.add_parser("probe-needles", help="Probe all needles and print raw LLM answers per needle")
    p2c.add_argument("--mode", required=True, choices=["full_context", "rag_sql_embeddings"])
    p2c.add_argument("--namespace", required=True)
    p2c.add_argument("--needles-file", required=True, help="TXT (one needle per line) or JSON list")
    p2c.add_argument("--questions-file", default="", help="Optional TXT/JSON questions list; if set, must match needles count")
    p2c.add_argument("--context-file", default="", help="Required for full_context mode")
    p2c.add_argument("--temperature", type=float, default=None, help="Optional override")
    p2c.add_argument("--run-prefix", default="needle_probe", help="Prefix for deterministic run ids")
    p2c.set_defaults(fn=cmd_probe_needles)

    p3 = sub.add_parser("debug-retrieve", help="Print top retrieved chunks for question")
    p3.add_argument("--namespace", required=True)
    p3.add_argument("--question", required=True)
    p3.set_defaults(fn=cmd_debug_retrieve)

    p4 = sub.add_parser("verify-logs", help="Print needle + model output from run logs")
    p4.add_argument("--runs-file", default="", help="Optional path to runs JSONL (defaults to NIAH_LOG_JSONL_PATH)")
    p4.add_argument("--needles-file", default="", help="Optional needles TXT/JSON to map needle_index -> needle text")
    p4.add_argument("--only-probe", action="store_true", help="Only include rows with metadata.probe_type=needle_probe")
    p4.add_argument("--last-n", type=int, default=0, help="Optional tail limit after filtering")
    p4.set_defaults(fn=cmd_verify_logs)

    args = ap.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
