from __future__ import annotations

import json
from typing import Iterable

import psycopg

from niah.models import RetrievedChunk


def _vec_sql_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"


def init_schema(*, dsn: str, dim: int) -> None:
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS niah_chunks (
                    namespace TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    meta_json TEXT NOT NULL DEFAULT '{{}}',
                    embedding vector({int(dim)}) NOT NULL,
                    ts BIGINT NOT NULL DEFAULT (EXTRACT(EPOCH FROM NOW()))::BIGINT,
                    PRIMARY KEY (namespace, doc_id, chunk_id)
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_niah_chunks_namespace ON niah_chunks(namespace)")
            conn.commit()


def upsert_chunks(
    *,
    dsn: str,
    dim: int,
    namespace: str,
    doc_id: str,
    rows: Iterable[tuple[str, str, list[float], dict]],
) -> int:
    count = 0
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for chunk_id, text, embedding, meta in rows:
                if len(embedding) != int(dim):
                    raise ValueError(f"Embedding dimension mismatch for {chunk_id}: {len(embedding)} != {int(dim)}")
                cur.execute(
                    f"""
                    INSERT INTO niah_chunks(namespace, doc_id, chunk_id, text, meta_json, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT(namespace, doc_id, chunk_id)
                    DO UPDATE SET text = EXCLUDED.text, meta_json = EXCLUDED.meta_json, embedding = EXCLUDED.embedding
                    """,
                    (namespace, doc_id, chunk_id, text, json.dumps(meta, ensure_ascii=False), _vec_sql_literal(embedding)),
                )
                count += 1
            conn.commit()
    return count


def search_chunks(
    *,
    dsn: str,
    dim: int,
    namespace: str,
    query_embedding: list[float],
    limit: int,
) -> list[RetrievedChunk]:
    if len(query_embedding) != int(dim):
        raise ValueError(f"Query embedding dimension mismatch: {len(query_embedding)} != {int(dim)}")
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT doc_id, chunk_id, text, meta_json, (embedding <=> %s::vector) AS distance
                FROM niah_chunks
                WHERE namespace = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (_vec_sql_literal(query_embedding), namespace, _vec_sql_literal(query_embedding), int(limit)),
            )
            out: list[RetrievedChunk] = []
            for row in cur.fetchall():
                out.append(
                    RetrievedChunk(
                        doc_id=str(row[0]),
                        chunk_id=str(row[1]),
                        text=str(row[2]),
                        meta_json=str(row[3] or "{}"),
                        distance=float(row[4]),
                    )
                )
            return out
