"""
sft_train.py — SecGPTv2.5 Stage 2: SFT on the v3 corpus.

Lesson from v2 (overfit: train 0.16 / val 2.18 at lr 1e-4): bigger model
gets a lower LR (5e-5) and val-loss-based best-checkpoint tracking.

Usage:
  python src/sft_train.py
  python src/sft_train.py --steps 10      # smoke test
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model import GPT, GPTConfig  # noqa: E402

STAGE1_OUT = ROOT / "stage1_pre-training" / "step5_training" / "output"
STAGE1_TOK = ROOT / "stage1_pre-training" / "step1_tokenizer" / "output"
SFT_OUT = ROOT / "stage2_sft" / "output"

BLOCK_SIZE = 512
BATCH_SIZE = 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 64)
    print("SecGPTv2.5 — Stage 2: SFT on v3 data")
    print("=" * 64)

    config = GPTConfig()
    model = GPT(config).to(device)
    ckpt = torch.load(STAGE1_OUT / "checkpoint_final.pt", map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"\n  Loaded pretrain checkpoint (step {ckpt['step']})")

    tokenizer = Tokenizer.from_file(str(STAGE1_TOK / "tokenizer.json"))
    sft_text = (SFT_OUT / "sft_corpus.txt").read_text(encoding="utf-8")
    data = torch.tensor(tokenizer.encode(sft_text).ids, dtype=torch.long)
    print(f"  SFT corpus: {len(data):,} tokens")

    split = int(len(data) * 0.95)
    train_data, val_data = data[:split], data[split:]

    def get_batch(split_data):
        ix = torch.randint(len(split_data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([split_data[i:i + BLOCK_SIZE] for i in ix])
        y = torch.stack([split_data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
        return x.to(device), y.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")

    best_val = float("inf")
    t0 = time.time()
    model.train()
    for step in range(args.steps + 1):
        if step % args.eval_interval == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                tl = [model(*get_batch(train_data))[1].item() for _ in range(15)]
                vl = [model(*get_batch(val_data))[1].item() for _ in range(15)]
            tv, vv = sum(tl) / 15, sum(vl) / 15
            marker = ""
            if vv < best_val:
                best_val = vv
                torch.save({"model": model.state_dict(), "config": config.__dict__,
                            "step": step}, SFT_OUT / "checkpoint_sft.pt")
                marker = " *best*"
            print(f"  step {step:>5} | train {tv:.4f} | val {vv:.4f} | "
                  f"{time.time()-t0:.0f}s{marker}", flush=True)
            model.train()

        if step == args.steps:
            break

        x, y = get_batch(train_data)
        with torch.amp.autocast("cuda"):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    log = {"steps": args.steps, "lr": args.lr, "best_val": round(best_val, 4),
           "total_min": round((time.time() - t0) / 60, 1),
           "sft_tokens": len(train_data)}
    with open(SFT_OUT / "sft_train_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  DONE — best val {best_val:.4f}, saved checkpoint_sft.pt")


if __name__ == "__main__":
    main()
