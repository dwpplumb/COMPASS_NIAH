from __future__ import annotations

import re


def clean_completion_text(text: str) -> str:
    out = str(text or "").strip()
    if not out:
        return out
    out = out.replace("```", "").strip()
    out = re.sub(r"^\*\*(.*?)\*\*$", r"\1", out, flags=re.DOTALL).strip()
    out = out.strip("*").strip()
    if "\n" in out:
        out = out.splitlines()[0].strip()
    out = out.strip("*").strip()
    out = out.strip('"').strip("'").strip()
    return out
