"""
build_dpo_corpus.py — SecGPTv2.5: preference pairs in tag format.

Same degradation strategies as the Qwen/GPT-2 lines (truncate / shuffle /
de-structure / hedge), formatted for the custom tokenizer:
  prompt  = "<|tag|>\\nQ: {instruction}\\nA:"
  chosen/rejected appended by the trainer.

Output: stage3_alignment/output/dpo_data.jsonl

Usage:
  python src/build_dpo_corpus.py
"""

import io
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
SFT_V3 = ROOT.parent / "SecGPT-Prod" / "data" / "v3" / "sft_v3.jsonl"
OUT = ROOT / "stage3_alignment" / "output" / "dpo_data.jsonl"

SEED = 42
CATEGORY_QUOTA = {"rule": 1500, "ttp": 600, "ref": 600, "kb": 300}
TAG_MAP = {"ttp": "ttp", "rule": "rule", "ref": "ref", "kb": "kb"}

HEDGES = ["I'm not entirely sure, but ", "This might not be fully accurate, but ",
          "It depends on many factors, but generally, ", "I think, though I could be wrong, "]


def degrade_truncate(text, rng):
    cut = max(1, int(len(text) * 0.4))
    return text[:cut].rsplit(" ", 1)[0].strip()


def degrade_shuffle(text, rng):
    parts = [p for p in re.split(r"(?<=[.!?])\s+|\n", text) if p.strip()]
    rng.shuffle(parts)
    return " ".join(parts)


def degrade_destructure(text, rng):
    flat = re.sub(r"^\s*[-*\d.]+\s*", "", text, flags=re.MULTILINE)
    flat = flat.replace("\n\n", " ").replace("\n", " ")
    return re.sub(r"\s{2,}", " ", flat).strip()


def degrade_hedge(text, rng):
    lines = text.split("\n")
    body = "\n".join(lines[1:]).strip() if len(lines) > 2 else text
    body = re.sub(r"\bT\d{4}(?:\.\d{3})?\b", "a certain technique", body)
    return rng.choice(HEDGES) + body[: len(body) // 2].strip()


DEGRADERS = [degrade_truncate, degrade_shuffle, degrade_destructure, degrade_hedge]


def main():
    rng = random.Random(SEED)
    by_cat = {c: [] for c in CATEGORY_QUOTA}
    with open(SFT_V3, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                if o["category"] in by_cat and 40 <= len(o["response"]) <= 800:
                    by_cat[o["category"]].append(o)

    pairs = []
    stats = Counter()
    for cat, quota in CATEGORY_QUOTA.items():
        pool = by_cat[cat]
        rng.shuffle(pool)
        for o in pool[:quota]:
            degrader = rng.choice(DEGRADERS)
            rejected = degrader(o["response"], rng)
            if len(rejected) < 20 or rejected == o["response"]:
                continue
            pairs.append({
                "prompt": f"<|{TAG_MAP[cat]}|>\nQ: {o['instruction']}\nA:",
                "chosen": o["response"],
                "rejected": rejected,
            })
            stats[cat] += 1

    rng.shuffle(pairs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  DPO pairs: {len(pairs):,} -> {OUT}")
    for cat, c in sorted(stats.items()):
        print(f"    {cat:8s} {c:,}")


if __name__ == "__main__":
    main()
