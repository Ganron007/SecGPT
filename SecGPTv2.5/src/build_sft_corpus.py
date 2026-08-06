"""
build_sft_corpus.py — SecGPTv2.5: SFT corpus from sft_v3.jsonl.

Same v3 data every model trains on, formatted for the tag-based custom
tokenizer:  <|tag|>\\nQ: {instruction}\\nA: {response}\\n\\n

Output: stage2_sft/output/sft_corpus.txt

Usage:
  python src/build_sft_corpus.py
"""

import io
import json
import random
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
SFT_V3 = ROOT.parent / "SecGPT-Prod" / "data" / "v3" / "sft_v3.jsonl"
OUT = ROOT / "stage2_sft" / "output" / "sft_corpus.txt"

TAG_MAP = {"ttp": "ttp", "rule": "rule", "ref": "ref", "kb": "kb",
           "classification": "kb"}
SEED = 42


def main():
    rng = random.Random(SEED)
    pairs = []
    with open(SFT_V3, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                tag = TAG_MAP.get(o["category"], "kb")
                pairs.append((tag, o["instruction"], o["response"]))
    rng.shuffle(pairs)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for tag, q, a in pairs:
            f.write(f"<|{tag}|>\nQ: {q}\nA: {a}\n\n")

    print(f"  SFT corpus: {len(pairs):,} pairs -> {OUT}")
    print(f"  Size: {OUT.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
