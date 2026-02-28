from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from niah.config import ensure_parent_dir
from niah.run_logger import now_ts


@dataclass(frozen=True)
class PromptIORecord:
    ts: int
    run_id: str
    mode: str
    namespace: str
    question: str
    system_text: str
    user_text: str
    prompt_text: str
    completion_text: str
    metadata: dict[str, Any]


def write_prompt_io_record(*, jsonl_path: str, rec: PromptIORecord) -> None:
    ensure_parent_dir(jsonl_path)
    line = json.dumps(rec.__dict__, ensure_ascii=False, sort_keys=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def make_prompt_io_record(
    *,
    run_id: str,
    mode: str,
    namespace: str,
    question: str,
    system_text: str,
    user_text: str,
    prompt_text: str,
    completion_text: str,
    metadata: dict[str, Any],
) -> PromptIORecord:
    return PromptIORecord(
        ts=now_ts(),
        run_id=run_id,
        mode=mode,
        namespace=namespace,
        question=question,
        system_text=system_text,
        user_text=user_text,
        prompt_text=prompt_text,
        completion_text=completion_text,
        metadata=dict(metadata),
    )
