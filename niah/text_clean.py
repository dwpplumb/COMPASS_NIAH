from __future__ import annotations

import re


def clean_completion_text(text: str) -> str:
    out = str(text or "").strip()
    if not out:
        return out
    out = re.sub(r"^\s*assistant\s*:\s*", "", out, flags=re.IGNORECASE).strip()
    out = out.replace("```", "").strip()
    out = re.sub(r"^\*\*(.*?)\*\*$", r"\1", out, flags=re.DOTALL).strip()
    out = out.strip("*").strip()
    out = out.strip("*").strip()
    out = out.strip('"').strip("'").strip()
    return out


def recover_full_sentence_from_context(*, answer_text: str, context_text: str) -> str:
    answer = str(answer_text or "").strip()
    context = str(context_text or "")
    if not answer or not context:
        return answer
    if answer.upper() == "UNSURE":
        return "UNSURE"

    def _norm(s: str) -> str:
        s = str(s or "").strip().strip('"').strip("'").strip()
        s = re.sub(r"\s+", " ", s)
        return s.lower()

    def _split_sentences(text: str) -> list[str]:
        parts: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            bits = re.split(r"(?<=[.!?])\s+", line)
            for bit in bits:
                bit = bit.strip()
                if bit:
                    parts.append(bit)
        return parts

    answer_norm = _norm(answer)
    candidates = _split_sentences(context)
    if not candidates:
        return answer

    # 1) exact sentence match
    for candidate in candidates:
        if _norm(candidate) == answer_norm:
            return candidate

    # 2) contains / prefix recovery for truncated answers
    if len(answer_norm) >= 12:
        contains = [c for c in candidates if answer_norm in _norm(c)]
        if contains:
            return min(contains, key=len)

    return answer
