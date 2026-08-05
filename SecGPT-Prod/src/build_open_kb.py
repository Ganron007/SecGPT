"""
build_open_kb.py — SecGPT-Prod v3: open-license security knowledge base chunks.

Clones open security knowledge repos and chunks their markdown by section.
These form the redistributable kb portion (replacement for proprietary CADRE).

Sources (git cloned into data/v3/open_src/, gitignored):
  - HackTricks (HackTricks-wiki/hacktricks)
  - OWASP Web Security Testing Guide (OWASP/wstg)

Output: data/v3/open_kb.jsonl

Usage:
  python src/build_open_kb.py
"""

import io
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "data" / "v3" / "open_src"
OUT = ROOT / "data" / "v3" / "open_kb.jsonl"

REPOS = {
    "hacktricks": "https://github.com/HackTricks-wiki/hacktricks.git",
    "owasp_wstg": "https://github.com/OWASP/wstg.git",
}

FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
HEADER_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)
MIN_CHUNK, MAX_CHUNK = 400, 1800


def clone_repos():
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in REPOS.items():
        dest = SRC_DIR / name
        if dest.exists():
            print(f"  {name}: already cloned")
            continue
        print(f"  Cloning {name} ...")
        r = subprocess.run(["git", "clone", "--depth", "1", url, str(dest)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  [WARN] clone failed for {name}: {r.stderr.strip()[:200]}")


def chunk_md(text):
    text = FRONTMATTER_RE.sub("", text, count=1)
    matches = list(HEADER_RE.finditer(text))
    if not matches:
        return []
    chunks = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = m.group(2).strip()
        body = text[m.end():end].strip()
        if len(body) < MIN_CHUNK:
            continue
        while len(body) > MAX_CHUNK:
            cut = body.rfind("\n\n", 0, MAX_CHUNK)
            if cut < MIN_CHUNK:
                cut = body.rfind(" ", MIN_CHUNK, MAX_CHUNK)
            if cut < MIN_CHUNK:
                break
            chunks.append((title, body[:cut].strip()))
            body = body[cut:].strip()
        if MIN_CHUNK <= len(body) <= MAX_CHUNK:
            chunks.append((title, body))
    return chunks


def main():
    print("=" * 64)
    print("SecGPT-Prod v3 — Open security KB chunks")
    print("=" * 64)

    clone_repos()

    records = []
    stats = Counter()
    for name in REPOS:
        src = SRC_DIR / name
        if not src.exists():
            continue
        for fp in src.rglob("*.md"):
            if any(part.startswith(".") or part in ("node_modules",) for part in fp.parts):
                continue
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if len(text) < MIN_CHUNK:
                continue
            for title, chunk in chunk_md(text):
                records.append({"source": name, "title": title or fp.stem, "text": chunk})
                stats[name] += 1

    seen = set()
    unique = []
    for r in records:
        key = re.sub(r"\s+", " ", r["text"][:200].lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    with open(OUT, "w", encoding="utf-8") as f:
        for r in unique:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 64}")
    for name, n in stats.items():
        print(f"    {name:15s} {n:,} chunks")
    print(f"  Open KB: {len(unique):,} chunks after dedup -> {OUT}")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
