from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_ESSAYS = [
    ("essay1", "The Age of the Essay", "https://paulgraham.com/essay.html"),
    ("essay2", "What You Can't Say", "https://paulgraham.com/say.html"),
    ("essay3", "The Bus Ticket Theory of Genius", "https://paulgraham.com/genius.html"),
    ("essay4", "Write Like You Talk", "https://paulgraham.com/talk.html"),
    ("essay5", "Maker's Schedule, Manager's Schedule", "https://paulgraham.com/makerschedule.html"),
    ("essay6", "Good and Bad Procrastination", "https://paulgraham.com/procrastination.html"),
    ("essay7", "The Top Idea in Your Mind", "https://paulgraham.com/top.html"),
    ("essay8", "Holding a Program in One's Head", "https://paulgraham.com/head.html"),
    ("essay9", "Startup = Growth", "https://paulgraham.com/growth.html"),
    ("essay10", "The Acceleration of Addictiveness", "https://paulgraham.com/addict.html"),
    ("essay11", "The Anatomy of Determination", "https://paulgraham.com/determ.html"),
    ("essay12", "What Kate Saw in Silicon Valley", "https://paulgraham.com/kate.html"),
    ("essay13", "The Origins of Wokeness", "https://paulgraham.com/woke.html"),
    ("essay14", "Writes and Write-Nots", "https://paulgraham.com/writes.html"),
    ("essay15", "When To Do What You Love", "https://paulgraham.com/love.html"),
]


def _normalize_text(raw: str) -> str:
    txt = re.sub(r"\r\n?", "\n", str(raw or ""))
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip() + "\n"


def _extract_main_text_from_html(html: str) -> str:
    # Dependency kept optional; script still works in local-file mode without bs4.
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    # Paul Graham pages are simple; body text extraction is sufficient for benchmark corpus building.
    text = soup.get_text("\n")
    return _normalize_text(text)


def _download(url: str, timeout_s: float) -> str:
    import requests

    resp = requests.get(url, timeout=float(timeout_s))
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code} for {url}")
    return resp.text


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare Paul Graham essay corpus for NIAH benchmark.")
    ap.add_argument("--out-dir", default="data/pg_essays", help="Output directory for essay text files.")
    ap.add_argument(
        "--mode",
        choices=["from_urls", "from_local"],
        default="from_local",
        help="from_local expects files in --local-dir (essay1.txt...); from_urls downloads from paulgraham.com URLs.",
    )
    ap.add_argument("--local-dir", default="data/pg_essays_raw", help="Local source directory for from_local mode.")
    ap.add_argument("--timeout-s", type=float, default=60.0)
    ap.add_argument(
        "--manifest-out",
        default="data/pg_essays/manifest.json",
        help="Write manifest (id,title,source_url,path).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    local_dir = Path(args.local_dir)

    manifest: list[dict] = []
    for essay_id, title, url in DEFAULT_ESSAYS:
        out_path = out_dir / f"{essay_id}.txt"
        if args.mode == "from_urls":
            html = _download(url=url, timeout_s=float(args.timeout_s))
            text = _extract_main_text_from_html(html)
        else:
            in_path = local_dir / f"{essay_id}.txt"
            if not in_path.exists():
                raise FileNotFoundError(f"Missing local essay file: {in_path}")
            text = _normalize_text(in_path.read_text(encoding="utf-8", errors="ignore"))
        out_path.write_text(text, encoding="utf-8")
        manifest.append(
            {
                "id": essay_id,
                "title": title,
                "source_url": url,
                "path": str(out_path).replace("\\", "/"),
                "chars": len(text),
                "words_est": len(text.split()),
            }
        )

    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "mode": args.mode,
                "out_dir": str(out_dir),
                "essays": len(manifest),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
