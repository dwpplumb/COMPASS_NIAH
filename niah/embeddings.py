from __future__ import annotations

import hashlib
import json
from typing import Any

from niah.config import AppConfig


def _deterministic_vector(text: str, dim: int) -> list[float]:
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    out: list[float] = []
    cur = seed
    while len(out) < dim:
        cur = hashlib.sha256(cur).digest()
        for i in range(0, len(cur), 4):
            if len(out) >= dim:
                break
            chunk = cur[i : i + 4]
            if len(chunk) < 4:
                break
            val = int.from_bytes(chunk, "big", signed=False)
            out.append((val / 4294967295.0) * 2.0 - 1.0)
    return out


def _openai_compat_embed(*, cfg: AppConfig, texts: list[str]) -> list[list[float]]:
    import requests

    if not cfg.embeddings_endpoint_url:
        raise RuntimeError("Missing NIAH_EMBEDDINGS_ENDPOINT_URL")
    if not cfg.embeddings_api_key:
        raise RuntimeError("Missing NIAH_EMBEDDINGS_API_KEY")
    if not cfg.embeddings_model:
        raise RuntimeError("Missing NIAH_EMBEDDINGS_MODEL")
    body: dict[str, Any] = {"model": cfg.embeddings_model, "input": texts}
    headers = {
        "Authorization": f"Bearer {cfg.embeddings_api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        cfg.embeddings_endpoint_url,
        headers=headers,
        data=json.dumps(body),
        timeout=float(cfg.embeddings_timeout_s),
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Embeddings HTTP {resp.status_code}: {resp.text[:400]}")
    obj = resp.json()
    data = obj.get("data")
    if not isinstance(data, list):
        raise RuntimeError("Invalid embeddings response: missing data[]")
    vectors: list[list[float]] = []
    for row in data:
        emb = row.get("embedding") if isinstance(row, dict) else None
        if not isinstance(emb, list):
            raise RuntimeError("Invalid embeddings response: embedding not list")
        vectors.append([float(x) for x in emb])
    return vectors


def _tei_embed(*, cfg: AppConfig, texts: list[str]) -> list[list[float]]:
    import requests

    if not cfg.embeddings_endpoint_url:
        raise RuntimeError("Missing NIAH_EMBEDDINGS_ENDPOINT_URL")
    vectors: list[list[float]] = []
    for text in texts:
        resp = requests.post(
            cfg.embeddings_endpoint_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps({"inputs": text}),
            timeout=float(cfg.embeddings_timeout_s),
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"TEI HTTP {resp.status_code}: {resp.text[:400]}")
        obj = resp.json()
        if isinstance(obj, list) and obj and isinstance(obj[0], (int, float)):
            vectors.append([float(x) for x in obj])
        elif isinstance(obj, list) and obj and isinstance(obj[0], list):
            vectors.append([float(x) for x in obj[0]])
        else:
            raise RuntimeError("Unexpected TEI response shape")
    return vectors


def embed_texts(*, cfg: AppConfig, texts: list[str]) -> list[list[float]]:
    provider = str(cfg.embeddings_provider or "").strip().lower()
    if provider in {"deterministic", "stub", ""}:
        return [_deterministic_vector(t, int(cfg.embeddings_dim)) for t in texts]
    if provider in {"openai_compat", "xai"}:
        return _openai_compat_embed(cfg=cfg, texts=texts)
    if provider == "tei":
        return _tei_embed(cfg=cfg, texts=texts)
    raise RuntimeError(f"Unsupported embeddings provider: {provider}")
