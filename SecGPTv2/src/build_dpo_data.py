"""
build_dpo_data.py — Phase A Stage 3: Build preference pairs for DPO alignment.

Creates (prompt, chosen, rejected) triplets:
  - chosen: correct structured answer (from SFT data)
  - rejected: degraded version (wrong format, incomplete, garbled)

Usage:
  python src/build_dpo_data.py
"""

import io
import json
import random
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
SFT_DATA = ROOT / "stage2_sft" / "output" / "sft_data.jsonl"
OUTPUT = ROOT / "stage3_alignment" / "output"

SEED = 42
MAX_PAIRS = 400


def degrade_response(text, rng):
    strategy = rng.choice(["truncate", "shuffle_sentences", "remove_structure", "add_hedging"])

    if strategy == "truncate":
        words = text.split()
        cut = rng.randint(len(words) // 4, len(words) // 2)
        return " ".join(words[:cut]) + "..."

    elif strategy == "shuffle_sentences":
        sentences = [s.strip() for s in text.replace("\n", ". ").split(".") if s.strip()]
        if len(sentences) > 2:
            rng.shuffle(sentences)
        return ". ".join(sentences) + "."

    elif strategy == "remove_structure":
        lines = text.split("\n")
        stripped = []
        for line in lines:
            line = line.strip()
            for prefix in ["ID:", "Description:", "Status:", "Level:", "Author:", "Detection Logic:", "References:"]:
                line = line.replace(prefix, "")
            stripped.append(line.strip())
        return " ".join(s for s in stripped if s)

    elif strategy == "add_hedging":
        hedges = [
            "I'm not sure but maybe ",
            "I think possibly ",
            "It could be that ",
            "I don't really know but ",
            "Perhaps incorrectly ",
        ]
        sentences = text.split("\n")
        result = []
        for s in sentences:
            if s.strip() and rng.random() < 0.4:
                result.append(rng.choice(hedges) + s.strip().lower())
            else:
                result.append(s)
        return "\n".join(result)

    return text


def main():
    rng = random.Random(SEED)
    print("=" * 60)
    print("Phase A — Stage 3: Building DPO Preference Pairs")
    print("=" * 60)

    sft_pairs = []
    with open(SFT_DATA, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sft_pairs.append(json.loads(line))

    print(f"\n  Loaded {len(sft_pairs)} SFT pairs")
    print(f"  Creating preference pairs (chosen vs rejected)...")

    dpo_pairs = []
    for pair in sft_pairs[:MAX_PAIRS]:
        prompt = f"<|{pair['tag']}|>\nQ: {pair['instruction']}\nA:"
        chosen = pair["response"]
        rejected = degrade_response(chosen, rng)
        if rejected != chosen and len(rejected) > 20:
            dpo_pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "tag": pair["tag"],
            })

    rng.shuffle(dpo_pairs)
    print(f"  Created {len(dpo_pairs)} preference pairs")

    from collections import Counter
    tag_dist = Counter(p["tag"] for p in dpo_pairs)
    for tag, count in tag_dist.most_common():
        print(f"    {tag:10s} {count}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / "dpo_data.jsonl"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for p in dpo_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n  Written: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

    print(f"\n  Example pair:")
    ex = dpo_pairs[0]
    print(f"    Prompt:   {ex['prompt'][:80]}...")
    print(f"    Chosen:   {ex['chosen'][:80]}...")
    print(f"    Rejected: {ex['rejected'][:80]}...")

    print(f"\n{'=' * 60}")
    print(f"  DONE — {len(dpo_pairs)} DPO pairs ready")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
