from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMResponse:
    text: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    chunk_id: str
    text: str
    distance: float
    meta_json: str
