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
    run_id = new_run_id()
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
        )
    elif args.mode == "rag_sql_embeddings":
        out = ask_rag_sql_embeddings(
            cfg=cfg,
            namespace=args.namespace,
            question=args.question,
            run_id=run_id,
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
            },
            ensure_ascii=False,
        )
    )
    return 0


def cmd_debug_retrieve(args: argparse.Namespace) -> int:
    cfg = load_config()
    print(format_hits_for_stdout(cfg=cfg, namespace=args.namespace, question=args.question))
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
    p2.set_defaults(fn=cmd_ask)

    p3 = sub.add_parser("debug-retrieve", help="Print top retrieved chunks for question")
    p3.add_argument("--namespace", required=True)
    p3.add_argument("--question", required=True)
    p3.set_defaults(fn=cmd_debug_retrieve)

    args = ap.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
