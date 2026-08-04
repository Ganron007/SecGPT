"""
build_kb_v3.py — SecGPT-Prod: re-extract CADRE KB from raw markdown.

Raw source: G:\\knowledgebase\\doc_extract (67 collections, 2,281 .md files).
The old page-based chunks (median 2,215 chars) were 78% ineligible for our
length filters. This builder chunks by SECTION, titles chunks by their actual
header, drops navigation junk, validates technique IDs against the real
ATT&CK ID set, and dedups across collections.

Output: data/kb_v3.jsonl  (gitignored)
  {collection, file, title, text, technique_ids}

Usage:
  python src/build_kb_v3.py [--raw-root G:\\knowledgebase\\doc_extract]
"""

import argparse
import hashlib
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
DFIR_MITRE = ROOT.parent / "SecGPTv2" / "data" / "dfir_nexus_sources" / "mitre_attack.jsonl"
OUT = ROOT / "data" / "kb_v3.jsonl"

FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
HEADER_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)
TECH_ID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
PAGE_LINK_RE = re.compile(r"\]\(#page-")

MIN_CHUNK = 400
MAX_CHUNK = 2000
TARGET_MERGE = 800

JUNK_TITLE_RE = re.compile(
    r"^(contents?|index|glossary|appendix|about|introduction to this (course|book)|"
    r"table of contents|references|bibliography|copyright|preface|acknowledg)", re.IGNORECASE)


def load_valid_ids():
    ids = set()
    if DFIR_MITRE.exists():
        for line in DFIR_MITRE.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                try:
                    ids.update(TECH_ID_RE.findall(json.loads(line).get("text", "")))
                except json.JSONDecodeError:
                    continue
    return ids


def split_sections(text):
    parts = []
    matches = list(HEADER_RE.finditer(text))
    if not matches:
        return [("", text)]
    if matches[0].start() > 0:
        parts.append(("", text[: matches[0].start()]))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts.append((m.group(2).strip(), text[m.end():end]))
    return parts


def chunk_text(title, body):
    body = body.strip()
    if not body:
        return []
    if len(body) <= MAX_CHUNK:
        return [(title, body)] if len(body) >= MIN_CHUNK else [(title, body)]
    chunks = []
    paragraphs = re.split(r"\n\s*\n", body)
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= MAX_CHUNK:
            current = (current + "\n\n" + para).strip()
        else:
            if len(current) >= MIN_CHUNK:
                chunks.append((title, current))
            current = para.strip()
    if len(current) >= MIN_CHUNK:
        chunks.append((title, current))
    return chunks


def is_junk(title, text):
    head = text[:300]
    if JUNK_TITLE_RE.match(title or ""):
        return True
    if head.count("](#") + len(PAGE_LINK_RE.findall(head)) >= 3:
        return True
    if text.strip().startswith("- [") and text.count("- [") >= 3:
        return True
    if len(re.sub(r"[\W_]+", "", text)) < 100:
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=r"G:\knowledgebase\doc_extract")
    args = parser.parse_args()
    raw_root = Path(args.raw_root)

    print("=" * 64)
    print("SecGPT-Prod — KB v3 re-extraction (section-based)")
    print("=" * 64)

    valid_ids = load_valid_ids()
    print(f"\n  Valid ATT&CK IDs loaded: {len(valid_ids)}")

    files = sorted(raw_root.rglob("*.md"))
    print(f"  Markdown files: {len(files):,}")

    chunks = []
    stats = Counter()
    for fp in files:
        collection = fp.relative_to(raw_root).parts[0]
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            stats["files_skipped"] += 1
            continue
        text = FRONTMATTER_RE.sub("", text, count=1)
        sections = split_sections(text)
        merged = []
        carry_title, carry_body = "", ""
        for title, body in sections:
            body = body.strip()
            if carry_body and len(carry_body) < TARGET_MERGE:
                carry_body = (carry_body + "\n\n" + (f"### {title}\n" if title else "") + body).strip()
                if not carry_title:
                    carry_title = title
                continue
            if carry_body:
                merged.append((carry_title, carry_body))
            carry_title, carry_body = title, body
        if carry_body:
            merged.append((carry_title, carry_body))

        for title, body in merged:
            for chunk_title, chunk in chunk_text(title, body):
                stats["raw_chunks"] += 1
                if is_junk(chunk_title, chunk):
                    stats["junk_dropped"] += 1
                    continue
                final_title = chunk_title or fp.stem
                chunks.append({
                    "collection": collection,
                    "file": fp.stem,
                    "title": final_title,
                    "text": chunk,
                })

    print(f"\n  Raw chunks: {stats['raw_chunks']:,}")
    print(f"  Junk dropped: {stats['junk_dropped']:,}")

    seen = set()
    unique = []
    for c in chunks:
        key = hashlib.md5(re.sub(r"\s+", " ", c["text"][:200].lower()).encode()).hexdigest()
        if key in seen:
            stats["dupes_dropped"] += 1
            continue
        seen.add(key)
        unique.append(c)
    print(f"  Cross-collection dupes dropped: {stats['dupes_dropped']:,}")

    n_ids = 0
    n_bad_ids = 0
    for c in unique:
        found = TECH_ID_RE.findall(c["text"])
        good = [i for i in dict.fromkeys(found) if i in valid_ids]
        bad = [i for i in dict.fromkeys(found) if i not in valid_ids]
        c["technique_ids"] = good
        n_ids += len(good)
        n_bad_ids += len(bad)
        del c["file"]

    lens = sorted(len(c["text"]) for c in unique)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for c in unique:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 64}")
    print(f"  KB v3: {len(unique):,} chunks -> {OUT}")
    if lens:
        print(f"  chunk len p10/p50/p90/p99: {lens[len(lens)//10]} / "
              f"{lens[len(lens)//2]} / {lens[int(len(lens)*.9)]} / {lens[int(len(lens)*.99)]}")
    print(f"  validated technique IDs kept: {n_ids:,} (garbage dropped: {n_bad_ids:,})")
    top = Counter(c["collection"] for c in unique).most_common(10)
    for name, n in top:
        print(f"    {name[:50]:52s} {n:,}")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
