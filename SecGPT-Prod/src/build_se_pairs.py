"""
build_se_pairs.py — SecGPT-Prod v3: real Q&A from Security StackExchange.

The only non-synthetic instruction data in the corpus: real questions with
community-accepted answers (CC BY-SA licensed).

Input: security.stackexchange.com Posts.xml from the StackExchange data dump
  (archive.org/details/stackexchange -> security.stackexchange.com.7z,
  ~1 GB). Extract and place at: data/v3/stackexchange/Posts.xml

Filters: questions with an accepted answer, question score >= 2, answer
score >= 3, lengths bounded, HTML stripped, deduped.

Output: data/v3/se_pairs.jsonl

Usage:
  python src/build_se_pairs.py [--posts data/v3/stackexchange/Posts.xml]
"""

import argparse
import html
import io
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "v3" / "se_pairs.jsonl"
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s{2,}")


def strip_html(text):
    text = TAG_RE.sub(" ", text or "")
    return WS_RE.sub(" ", html.unescape(text)).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posts", default=str(ROOT / "data" / "v3" / "stackexchange" / "Posts.xml"))
    parser.add_argument("--max-pairs", type=int, default=8000)
    args = parser.parse_args()
    posts_path = Path(args.posts)

    print("=" * 64)
    print("SecGPT-Prod v3 — Security StackExchange Q&A pairs")
    print("=" * 64)

    if not posts_path.exists():
        print(f"\n  Posts.xml not found at {posts_path}")
        print("  Download: archive.org/details/stackexchange -> security.stackexchange.com.7z")
        print("  Extract Posts.xml into data/v3/stackexchange/ and rerun.")
        sys.exit(1)

    print(f"\n  Parsing {posts_path} ({posts_path.stat().st_size / 1e6:.0f} MB) ...")
    questions = {}
    answers = {}
    n_rows = 0
    for _, elem in ET.iterparse(str(posts_path), events=("end",)):
        if elem.tag != "row":
            continue
        n_rows += 1
        pt = elem.get("PostTypeId")
        score = int(elem.get("Score", "0"))
        if pt == "1":
            acc = elem.get("AcceptedAnswerId")
            if acc and score >= 2:
                questions[elem.get("Id")] = {
                    "title": strip_html(elem.get("Title", "")),
                    "body": strip_html(elem.get("Body", "")),
                    "accepted": acc,
                }
        elif pt == "2" and score >= 3:
            answers[elem.get("Id")] = strip_html(elem.get("Body", ""))
        elem.clear()
    print(f"  rows scanned: {n_rows:,} | candidate questions: {len(questions):,} | scored answers: {len(answers):,}")

    pairs = []
    seen = set()
    for q in questions.values():
        ans = answers.get(q["accepted"])
        if not ans or not q["title"]:
            continue
        if not (20 <= len(ans) <= 1800):
            continue
        instruction = q["title"] if len(q["title"]) >= 25 else f"{q['title']}\n{q['body'][:400]}"
        if len(instruction) > 800:
            instruction = instruction[:800].rsplit(" ", 1)[0]
        key = instruction.lower()[:100]
        if key in seen:
            continue
        seen.add(key)
        pairs.append({"instruction": instruction, "response": ans,
                      "category": "kb", "kind": "real_qa"})
        if len(pairs) >= args.max_pairs:
            break

    with open(OUT, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 64}")
    print(f"  SE pairs: {len(pairs):,} -> {OUT}")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
