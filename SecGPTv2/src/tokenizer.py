"""
tokenizer.py — SecGPT v2 Step 1: BPE tokenizer (8000 vocab).

Trains a BPE tokenizer on the corpus using HuggingFace tokenizers,
encodes the full corpus, splits train/val, saves all outputs.

Usage:
  python src/tokenizer.py
"""

import io
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
STEP1 = ROOT / "stage1_pre-training" / "step1_tokenizer"
INPUT = STEP1 / "input"
OUTPUT = STEP1 / "output"

CORPUS_FILE = ROOT / "stage1_pre-training" / "step0_corpus" / "output" / "corpus.txt"
VOCAB_SIZE = 8000
TRAIN_RATIO = 0.9
SEED = 42
SPECIAL_TOKENS = ["<|kb|>", "<|rule|>", "<|ttp|>", "<|ref|>", "<|spam|>", "<|ham|>", "<|net|>", "<|malware|>", "<|pad|>"]


def main():
    print("=" * 60)
    print("SecGPT v2 — Step 1: BPE Tokenizer (8000 vocab)")
    print("=" * 60)

    print(f"\n  Corpus: {CORPUS_FILE}")
    print(f"  Corpus size: {CORPUS_FILE.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Target vocab: {VOCAB_SIZE}")
    print(f"  Special tokens: {len(SPECIAL_TOKENS)}")

    print("\n[1/5] Training BPE tokenizer...")
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS + ["<|unk|>"],
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train([str(CORPUS_FILE)], trainer)
    actual_vocab = tokenizer.get_vocab_size()
    print(f"  Trained vocab size: {actual_vocab}")

    print("\n[2/5] Testing encode/decode...")
    test_str = "<|kb|>\nThe adversary used PowerShell to execute encoded commands."
    encoded = tokenizer.encode(test_str)
    decoded = tokenizer.decode(encoded.ids)
    print(f"  Input:   {test_str!r}")
    print(f"  Tokens:  {encoded.ids[:20]}...")
    print(f"  Pieces:  {encoded.tokens[:20]}...")
    print(f"  Decoded: {decoded!r}")
    assert "PowerShell" in decoded, "Round-trip lost content"
    print(f"  Round-trip: OK")

    test_rule = "<|rule|>\nrule Emotet { strings: $a = \"Invoke-Mimikatz\" condition: $a }"
    enc2 = tokenizer.encode(test_rule)
    print(f"\n  Rule test: {len(enc2.ids)} tokens for {len(test_rule)} chars")
    print(f"  Compression: {len(test_rule) / len(enc2.ids):.1f} chars/token")

    print("\n[3/5] Encoding full corpus...")
    corpus_text = CORPUS_FILE.read_text(encoding="utf-8")
    full_encoded = tokenizer.encode(corpus_text)
    total_tokens = len(full_encoded.ids)
    print(f"  Corpus chars:  {len(corpus_text):,}")
    print(f"  Corpus tokens: {total_tokens:,}")
    print(f"  Compression:   {len(corpus_text) / total_tokens:.2f} chars/token")

    data = torch.tensor(full_encoded.ids, dtype=torch.long)
    print(f"  Tensor: {data.shape} ({data.dtype})")
    print(f"  Token range: [{data.min().item()}, {data.max().item()}]")

    print("\n[4/5] Splitting train/val (90/10)...")
    torch.manual_seed(SEED)
    n = data.shape[0]
    split_idx = int(n * TRAIN_RATIO)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    print(f"  Train: {train_data.shape[0]:,} tokens ({train_data.shape[0] / n * 100:.1f}%)")
    print(f"  Val:   {val_data.shape[0]:,} tokens ({val_data.shape[0] / n * 100:.1f}%)")

    print("\n[5/5] Saving outputs...")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    INPUT.mkdir(parents=True, exist_ok=True)

    tokenizer_path = OUTPUT / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"  {tokenizer_path.name}: tokenizer model")

    train_path = OUTPUT / "train_data.bin"
    torch.save(train_data, train_path)
    print(f"  {train_path.name}: {train_path.stat().st_size / 1024 / 1024:.1f} MB")

    val_path = OUTPUT / "val_data.bin"
    torch.save(val_data, val_path)
    print(f"  {val_path.name}: {val_path.stat().st_size / 1024 / 1024:.1f} MB")

    import math
    meta = {
        "tokenizer_type": "BPE (ByteLevel)",
        "vocab_size": actual_vocab,
        "total_tokens": n,
        "train_tokens": train_data.shape[0],
        "val_tokens": val_data.shape[0],
        "train_ratio": TRAIN_RATIO,
        "seed": SEED,
        "corpus_chars": len(corpus_text),
        "compression_ratio": round(len(corpus_text) / total_tokens, 2),
        "block_size_suggestion": 256,
        "random_guess_loss": round(math.log(actual_vocab), 4),
        "special_tokens": SPECIAL_TOKENS,
    }
    meta_path = OUTPUT / "tokenizer_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  {meta_path.name}: metadata")

    print(f"\n{'=' * 60}")
    print(f"  DONE")
    print(f"  Vocab:    {actual_vocab} BPE tokens")
    print(f"  Tokens:   {n:,} total ({train_data.shape[0]:,} train / {val_data.shape[0]:,} val)")
    print(f"  Compress: {len(corpus_text) / total_tokens:.2f} chars/token (v1 was 1.0)")
    print(f"  Random-guess loss: ln({actual_vocab}) = {meta['random_guess_loss']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
