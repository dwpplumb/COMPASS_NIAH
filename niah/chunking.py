from __future__ import annotations


def chunk_text(*, text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    src = str(text or "")
    if not src.strip():
        return []
    c = max(200, int(chunk_chars))
    o = max(0, min(int(overlap_chars), c - 1))
    chunks: list[str] = []
    i = 0
    n = len(src)
    while i < n:
        j = min(n, i + c)
        piece = src[i:j].strip()
        if piece:
            chunks.append(piece)
        if j >= n:
            break
        i = max(i + 1, j - o)
    return chunks
