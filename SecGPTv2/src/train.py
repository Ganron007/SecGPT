"""
train.py — SecGPT v2 Step 5: Training loop with mixed precision + cosine LR.

Usage:
  python src/train.py
  python src/train.py --steps 10000 --lr 6e-4
"""

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

import torch

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import GPT, GPTConfig

STEP1_OUT = ROOT / "stage1_pre-training" / "step1_tokenizer" / "output"
STEP5_OUT = ROOT / "stage1_pre-training" / "step5_training" / "output"

BLOCK_SIZE = 256
BATCH_SIZE = 32


def get_batch(data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE, device="cuda"):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters=50, device="cuda"):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, device=device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(
    max_steps=20000,
    learning_rate=6e-4,
    min_lr=6e-5,
    weight_decay=0.1,
    warmup_steps=500,
    eval_interval=1000,
    eval_iters=50,
    checkpoint_interval=5000,
    seed=42,
):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("SecGPT v2 — Training")
    print("=" * 60)

    config = GPTConfig()
    model = GPT(config).to(device)
    print(f"\n  Model: {model.num_parameters():,} params ({model.num_parameters() / 1e6:.1f}M) on {device}")
    print(f"  Config: {config.n_layer}L x {config.n_head}H x {config.n_embd}d, block={config.block_size}")
    print(f"  Training: {max_steps} steps, lr={learning_rate} (cosine → {min_lr})")
    print(f"  Batch: {BATCH_SIZE} x {BLOCK_SIZE} = {BATCH_SIZE * BLOCK_SIZE:,} tokens/step")
    print(f"  Mixed precision: fp16")

    train_data = torch.load(STEP1_OUT / "train_data.bin", weights_only=True)
    val_data = torch.load(STEP1_OUT / "val_data.bin", weights_only=True)
    print(f"  Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda")

    STEP5_OUT.mkdir(parents=True, exist_ok=True)
    log = []

    print(f"\n{'Step':>6} | {'LR':>8} | {'Train':>8} | {'Val':>8} | {'Time':>7}")
    print("-" * 55)

    t0 = time.time()
    for step in range(max_steps + 1):
        lr = get_lr(step, warmup_steps, max_steps, learning_rate, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if step % eval_interval == 0 or step == max_steps:
            losses = estimate_loss(model, train_data, val_data, eval_iters, device)
            elapsed = time.time() - t0
            print(f"{step:>6} | {lr:>8.5f} | {losses['train']:>8.4f} | {losses['val']:>8.4f} | {elapsed:>6.1f}s")
            log.append({"step": step, "lr": round(lr, 6), "train_loss": round(losses["train"], 4), "val_loss": round(losses["val"], 4), "elapsed_s": round(elapsed, 1)})

        if step > 0 and step % checkpoint_interval == 0:
            torch.save({"model": model.state_dict(), "config": config.__dict__, "step": step}, STEP5_OUT / f"checkpoint_step{step}.pt")

        if step == max_steps:
            break

        x, y = get_batch(train_data, device=device)
        with torch.amp.autocast("cuda"):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    total_time = time.time() - t0
    print("-" * 55)
    print(f"  Complete in {total_time:.1f}s ({total_time / 60:.1f} min)")

    torch.save({"model": model.state_dict(), "config": config.__dict__, "step": max_steps}, STEP5_OUT / "checkpoint_final.pt")
    ckpt_size = (STEP5_OUT / "checkpoint_final.pt").stat().st_size / 1024 / 1024
    print(f"  Checkpoint: {ckpt_size:.1f} MB")

    with open(STEP5_OUT / "train_log.json", "w") as f:
        json.dump({"config": config.__dict__, "max_steps": max_steps, "lr": learning_rate, "total_time_s": round(total_time, 1), "device": device, "params": model.num_parameters(), "log": log}, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Final train: {log[-1]['train_loss']}")
    print(f"  Final val:   {log[-1]['val_loss']}")
    print(f"  Baseline:    {math.log(config.vocab_size):.4f} (ln {config.vocab_size})")
    print(f"  Time:        {total_time:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(max_steps=args.steps, learning_rate=args.lr, min_lr=args.min_lr, weight_decay=args.wd, warmup_steps=args.warmup, eval_interval=args.eval_interval, eval_iters=args.eval_iters, checkpoint_interval=args.checkpoint_interval, seed=args.seed)
