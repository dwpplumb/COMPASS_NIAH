from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """
    Lightweight, deterministic estimate.
    """
    if not text:
        return 0
    # Common heuristic: ~4 chars per token in English-like text.
    # Add small floor bias so tiny prompts are not underestimated.
    n = len(text)
    return max(1, int(n / 4))
