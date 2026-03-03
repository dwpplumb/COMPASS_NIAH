from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_POSITIONS = [0.10, 0.25, 0.50, 0.75, 0.90]


def _load_essay_paths(essays_dir: Path) -> list[Path]:
    files = sorted(essays_dir.glob("essay*.txt"), key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"No essay*.txt files found in {essays_dir}")
    return files


def _load_needles(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, list):
            raise ValueError("Needles JSON must be a list of strings.")
        return [str(x).strip() for x in obj if str(x).strip()]
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln]


def _words(text: str) -> list[str]:
    return text.split()


def _insert_by_word_pos(text: str, insert_text: str, pos_word_idx: int) -> str:
    ws = _words(text)
    idx = max(0, min(len(ws), int(pos_word_idx)))
    ins = _words(insert_text)
    out = ws[:idx] + ins + ws[idx:]
    return " ".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build large NIAH haystack from PG essays and inject needles.")
    ap.add_argument("--essays-dir", default="data/pg_essays")
    ap.add_argument("--needles-file", default="data/needles.txt", help="TXT (one needle/line) or JSON list.")
    ap.add_argument("--base-out", default="data/full_base_haystack.txt")
    ap.add_argument("--full-out", default="data/full_haystack_with_needles.txt")
    ap.add_argument("--manifest-out", default="data/haystack_manifest.json")
    ap.add_argument("--base-repeat", type=int, default=1, help="Repeat each essay block inside base haystack.")
    ap.add_argument("--global-repeat", type=int, default=6, help="Repeat base haystack to extend context.")
    ap.add_argument("--filler-repeat", type=int, default=5, help="Repeat first 1000 chars of each essay as local filler.")
    ap.add_argument("--global-filler-lines", type=int, default=1000)
    ap.add_argument("--positions", default="0.10,0.25,0.50,0.75,0.90")
    ap.add_argument("--with-needle-tags", action="store_true", help="Prefix inserted needles with [NIAH_NEEDLE_i].")
    args = ap.parse_args()

    essays_dir = Path(args.essays_dir)
    needles_path = Path(args.needles_file)
    essay_paths = _load_essay_paths(essays_dir)
    needles = _load_needles(needles_path)
    positions = [float(x.strip()) for x in str(args.positions).split(",") if x.strip()]
    if not positions:
        positions = list(DEFAULT_POSITIONS)

    blocks: list[str] = []
    for essay_path in essay_paths:
        essay_text = essay_path.read_text(encoding="utf-8", errors="ignore").strip()
        local_filler = (essay_text[:1000] + "\n") * max(0, int(args.filler_repeat))
        one = f"{essay_text}\n\n-----\n\n{local_filler}"
        blocks.extend([one] * max(1, int(args.base_repeat)))
    base_haystack = "\n".join(blocks).strip() + "\n"

    base_out = Path(args.base_out)
    base_out.parent.mkdir(parents=True, exist_ok=True)
    base_out.write_text(base_haystack, encoding="utf-8")

    global_filler = ("Filler text to extend context length. This is random noise.\n" * max(0, int(args.global_filler_lines))).strip()
    full_haystack = []
    for i in range(max(1, int(args.global_repeat))):
        full_haystack.append(base_haystack)
        if i < max(1, int(args.global_repeat)) - 1 and global_filler:
            full_haystack.append(global_filler)
    merged = "\n".join(full_haystack).strip()

    total_words_before = len(_words(merged))
    inserts: list[dict] = []
    for i, needle in enumerate(needles):
        p = positions[i % len(positions)]
        target_idx = int(total_words_before * p)
        inserted_needle = f"[NIAH_NEEDLE_{i+1}] {needle}" if bool(args.with_needle_tags) else needle
        merged = _insert_by_word_pos(merged, inserted_needle, target_idx)
        inserts.append({"needle_index": i + 1, "position_ratio": p, "target_word_idx": target_idx, "text": needle})

    full_out = Path(args.full_out)
    full_out.parent.mkdir(parents=True, exist_ok=True)
    full_out.write_text(merged, encoding="utf-8")

    manifest = {
        "ok": True,
        "essay_files": [str(p).replace("\\", "/") for p in essay_paths],
        "needles_file": str(needles_path).replace("\\", "/"),
        "base_out": str(base_out).replace("\\", "/"),
        "full_out": str(full_out).replace("\\", "/"),
        "base_chars": len(base_haystack),
        "base_words_est": len(_words(base_haystack)),
        "full_chars": len(merged),
        "full_words_est": len(_words(merged)),
        "full_tokens_est_4chars": int(len(merged) / 4),
        "inserts": inserts,
    }
    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("ok", "base_words_est", "full_words_est", "full_tokens_est_4chars", "full_out")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
