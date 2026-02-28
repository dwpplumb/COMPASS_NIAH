from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from niah.config import ensure_parent_dir


@dataclass(frozen=True)
class RunLogRecord:
    run_id: str
    ts: int
    mode: str
    namespace: str
    question: str
    input_chars: int
    output_chars: int
    prompt_tokens_est: int
    completion_tokens_est: int
    total_tokens_est: int
    latency_ms: int
    completion_text: str
    metadata: dict[str, Any]


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:16]}"


def write_record(*, jsonl_path: str, rec: RunLogRecord) -> None:
    ensure_parent_dir(jsonl_path)
    line = json.dumps(rec.__dict__, ensure_ascii=False, sort_keys=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def now_ts() -> int:
    return int(time.time())
