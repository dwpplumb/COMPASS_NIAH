from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_POSITIONS = [0.05, 0.25, 0.50, 0.75, 0.95]


def _load_needles(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, list):
            raise ValueError("Needles JSON must be a list.")
        return [str(x).strip() for x in obj if str(x).strip()]
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _insert_at_char_pos(text: str, needle: str, ratio: float) -> tuple[str, int]:
    r = max(0.0, min(1.0, float(ratio)))
    pos = int(len(text) * r)
    return text[:pos] + needle + text[pos:], pos


def main() -> int:
    ap = argparse.ArgumentParser(description="Create systematic NIAH cases with fixed needle positions.")
    ap.add_argument("--base-file", default="data/full_base_haystack.txt")
    ap.add_argument("--needles-file", default="data/needles.txt")
    ap.add_argument("--out-dir", default="data/cases")
    ap.add_argument("--positions", default="0.05,0.25,0.50,0.75,0.95")
    ap.add_argument("--tag-prefix", default="NIAH_NEEDLE")
    ap.add_argument("--with-needle-tags", action="store_true", help="Prefix inserted needles with [TAG_i].")
    args = ap.parse_args()

    base_path = Path(args.base_file)
    needles_path = Path(args.needles_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = base_path.read_text(encoding="utf-8", errors="ignore")
    needles = _load_needles(needles_path)
    pos_values = [float(x.strip()) for x in str(args.positions).split(",") if x.strip()]
    if not pos_values:
        pos_values = list(DEFAULT_POSITIONS)

    manifest_rows: list[dict] = []
    for i, needle in enumerate(needles, start=1):
        for p in pos_values:
            inserted = f"[{args.tag_prefix}_{i}] {needle}" if bool(args.with_needle_tags) else needle
            injected, char_pos = _insert_at_char_pos(base, inserted, p)
            pct_label = int(round(p * 100))
            out_file = out_dir / f"case_n{i:02d}_p{pct_label:02d}.txt"
            out_file.write_text(injected, encoding="utf-8")
            manifest_rows.append(
                {
                    "needle_index": i,
                    "needle_text": needle,
                    "position_ratio": p,
                    "position_pct": pct_label,
                    "char_pos": char_pos,
                    "file": str(out_file).replace("\\", "/"),
                    "question_hint": f"What is needle {i}?",
                }
            )

    manifest_path = out_dir / "cases_manifest.json"
    manifest_path.write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "cases": len(manifest_rows),
                "positions": pos_values,
                "needles": len(needles),
                "manifest": str(manifest_path).replace("\\", "/"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
