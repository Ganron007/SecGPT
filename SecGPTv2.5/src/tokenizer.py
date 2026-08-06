"""
tokenizer.py — SecGPTv2.5 Step 1: BPE tokenizer (16,384 vocab).

v2's 8K vocab under-compressed security text; 16K halves the token count for
the same content. Trains on the 333 MB v3-era corpus, encodes, splits 90/10.

Usage:
  python src/tokenizer.py
"""

import io
import json
import math
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
STEP1 = ROOT / "stage1_pre-training" / "step1_tokenizer"
OUTPUT = STEP1 / "output"
CORPUS_FILE = ROOT / "stage1_pre-training" / "step0_corpus" / "output" / "corpus.txt"

VOCAB_SIZE = 16384
TRAIN_RATIO = 0.9
SEED = 42
SPECIAL_TOKENS = ["<|kb|>", "<|rule|>", "<|ttp|>", "<|ref|>", "<|spam|>",
                  "<|ham|>", "<|net|>", "<|pad|>"]


def main():
    print("=" * 64)
    print(f"SecGPTv2.5 — Step 1: BPE tokenizer ({VOCAB_SIZE:,} vocab)")
    print("=" * 64)
    print(f"\n  Corpus: {CORPUS_FILE.stat().st_size / 1e6:.0f} MB")

    print("\n  [1/4] Training BPE ...")
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE,
                                  special_tokens=SPECIAL_TOKENS + ["<|unk|>"],
                                  min_frequency=3, show_progress=True)
    tokenizer.train([str(CORPUS_FILE)], trainer)
    vocab = tokenizer.get_vocab_size()
    print(f"  vocab: {vocab:,}")

    print("\n  [2/4] Round-trip test ...")
    test = "<|kb|>\nThe adversary used PowerShell to execute encoded commands."
    decoded = tokenizer.decode(tokenizer.encode(test).ids)
    assert "PowerShell" in decoded, "round-trip failed"
    print("  OK")

    print("\n  [3/4] Encoding corpus ...")
    corpus_text = CORPUS_FILE.read_text(encoding="utf-8")
    ids = tokenizer.encode(corpus_text).ids
    n = len(ids)
    print(f"  {len(corpus_text):,} chars -> {n:,} tokens "
          f"({len(corpus_text) / n:.2f} chars/token)")
    data = torch.tensor(ids, dtype=torch.long)
    del ids

    torch.manual_seed(SEED)
    split = int(n * TRAIN_RATIO)
    train_data, val_data = data[:split], data[split:]
    print(f"  train {train_data.shape[0]:,} / val {val_data.shape[0]:,}")

    print("\n  [4/4] Saving ...")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(OUTPUT / "tokenizer.json"))
    torch.save(train_data, OUTPUT / "train_data.bin")
    torch.save(val_data, OUTPUT / "val_data.bin")
    meta = {"vocab_size": vocab, "total_tokens": n,
            "train_tokens": train_data.shape[0], "val_tokens": val_data.shape[0],
            "corpus_chars": len(corpus_text),
            "compression": round(len(corpus_text) / n, 2),
            "random_guess_loss": round(math.log(vocab), 4),
            "special_tokens": SPECIAL_TOKENS}
    with open(OUTPUT / "tokenizer_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  DONE — {n:,} tokens, random-guess loss {meta['random_guess_loss']}")


if __name__ == "__main__":
    main()
