"""
build_dpo_data.py — SecGPT-Prod Stage 2: DPO preference pairs.

Chosen  = real SFT response (structured).
Rejected = degraded variant via 4 strategies (truncate / shuffle /
           de-structure / hedge) — the v2 approach, scaled.

Skips classification (labels are short by nature). Emphasizes rule/ttp/ref
where structure matters most. Output: data/dpo_4k.jsonl (gitignored).

Usage:
  python src/build_dpo_data.py [--pairs 4000]
"""

import argparse
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
SFT_FILE = ROOT / "data" / "sft_32k.jsonl"
OUT_FILE = ROOT / "data" / "dpo_4k.jsonl"
SEED = 42

CATEGORY_QUOTA = {"rule": 2000, "ttp": 800, "ref": 800, "kb": 400}

HEDGES = [
    "I'm not entirely sure, but ",
    "This might not be fully accurate, but ",
    "It depends on many factors, but generally, ",
    "I think, though I could be wrong, ",
]


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=4000)
    args = parser.parse_args()

    rng = random.Random(SEED)
    print("=" * 60)
    print("SecGPT-Prod — Build DPO Preference Pairs")
    print("=" * 60)

    by_cat = {c: [] for c in CATEGORY_QUOTA}
    with open(SFT_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if o["category"] in by_cat and 40 <= len(o["response"]) <= 800:
                by_cat[o["category"]].append(o)

    pairs = []
    stats = Counter()
    for cat, quota in CATEGORY_QUOTA.items():
        pool = by_cat[cat]
        rng.shuffle(pool)
        n = min(quota, len(pool), max(0, args.pairs - len(pairs)))
        for o in pool[:n]:
            degrader = rng.choice(DEGRADERS)
            rejected = degrader(o["response"], rng)
            if len(rejected) < 20 or rejected == o["response"]:
                continue
            pairs.append({
                "prompt": f"<|user|>\n{o['instruction']}<|end|>\n<|assistant|>\n",
                "chosen": o["response"] + "<|end|>",
                "rejected": rejected + "<|end|>",
                "category": cat,
                "strategy": degrader.__name__.replace("degrade_", ""),
            })
            stats[cat] += 1
            stats[degrader.__name__] += 1

    rng.shuffle(pairs)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n  Pairs written: {len(pairs)} -> {OUT_FILE}")
    for cat in CATEGORY_QUOTA:
        print(f"    {cat:15s} {stats[cat]}")
    for d in DEGRADERS:
        print(f"    {d.__name__.replace('degrade_', ''):15s} {stats[d.__name__]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
