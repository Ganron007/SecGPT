"""
sft_train.py — Phase A Stage 2: Supervised Fine-Tuning on SecGPT v2.

Fine-tunes the pretrained checkpoint on Q&A pairs so the model learns
to ANSWER questions instead of just completing text.

Key difference from pretraining:
  - Pretraining: predict next token on raw corpus (all tokens contribute to loss)
  - SFT: predict next token on Q&A format (model learns Q: → A: pattern)

Usage:
  python src/sft_train.py
  python src/sft_train.py --steps 2000 --lr 1e-4
"""

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model import GPT, GPTConfig

STAGE1_OUT = ROOT / "stage1_pre-training" / "step5_training" / "output"
STAGE1_TOK = ROOT / "stage1_pre-training" / "step1_tokenizer" / "output"
SFT_OUT = ROOT / "stage2_sft" / "output"

BLOCK_SIZE = 256
BATCH_SIZE = 8


def main():
    parser = argparse.ArgumentParser(description="SecGPT v2 SFT")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Phase A — Stage 2: Supervised Fine-Tuning")
    print("=" * 60)

    config = GPTConfig()
    model = GPT(config).to(device)
    ckpt = torch.load(STAGE1_OUT / "checkpoint_final.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"\n  Loaded pretrained checkpoint (step {ckpt['step']})")
    print(f"  Model: {model.num_parameters():,} params on {device}")

    tokenizer = Tokenizer.from_file(str(STAGE1_TOK / "tokenizer.json"))
    sft_corpus = (SFT_OUT / "sft_corpus.txt").read_text(encoding="utf-8")
    encoded = tokenizer.encode(sft_corpus)
    data = torch.tensor(encoded.ids, dtype=torch.long)
    print(f"  SFT corpus: {len(sft_corpus):,} chars → {len(data):,} tokens")
    print(f"  Training: {args.steps} steps, lr={args.lr}, batch={BATCH_SIZE}, block={BLOCK_SIZE}")

    split = int(len(data) * 0.9)
    train_data = data[:split]
    val_data = data[split:]
    print(f"  Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")

    def get_batch(split_data):
        ix = torch.randint(len(split_data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([split_data[i:i + BLOCK_SIZE] for i in ix])
        y = torch.stack([split_data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
        return x.to(device), y.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")

    print(f"\n{'Step':>6} | {'Train':>8} | {'Val':>8} | {'Time':>7}")
    print("-" * 45)

    t0 = time.time()
    model.train()
    for step in range(args.steps + 1):
        if step % args.eval_interval == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                train_losses = [model(*get_batch(train_data))[1].item() for _ in range(20)]
                val_losses = [model(*get_batch(val_data))[1].item() for _ in range(20)]
            elapsed = time.time() - t0
            print(f"{step:>6} | {sum(train_losses)/20:>8.4f} | {sum(val_losses)/20:>8.4f} | {elapsed:>6.1f}s")
            model.train()

        if step == args.steps:
            break

        x, y = get_batch(train_data)
        with torch.amp.autocast("cuda"):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    total = time.time() - t0
    out_path = SFT_OUT / "checkpoint_sft.pt"
    torch.save({"model": model.state_dict(), "config": config.__dict__, "step": args.steps, "base_step": ckpt["step"]}, out_path)
    print(f"\n  SFT complete in {total:.1f}s")
    print(f"  Checkpoint: {out_path.name} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    log = {"steps": args.steps, "lr": args.lr, "total_time_s": round(total, 1), "sft_tokens": len(train_data), "base_checkpoint": "stage1 step 20000"}
    with open(SFT_OUT / "sft_train_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  DONE — model now knows Q: → A: pattern")
    print(f"  Test: python src/generate.py --checkpoint stage2_sft/output/checkpoint_sft.pt --prompt \"<|ttp|>\\nQ: What is T1059?\\nA:\"")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
