"""
train.py — SecGPTv2.5: pretraining loop (fp16 AMP, cosine LR, resume).

~108.7M tokens, batch 16 x 512 = 8,192 tokens/step. Default 24,000 steps
(~1.8 epochs). Checkpoints every 2,000 steps (last 3 kept) with full
optimizer state for clean resume.

Usage:
  python src/train.py                    # full run
  python src/train.py --steps 10         # smoke test
  python src/train.py --resume           # resume from latest checkpoint
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
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import GPT, GPTConfig  # noqa: E402

STEP1_OUT = ROOT / "stage1_pre-training" / "step1_tokenizer" / "output"
STEP5_OUT = ROOT / "stage1_pre-training" / "step5_training" / "output"

BLOCK_SIZE = 512
BATCH_SIZE = 16


def get_batch(data, device, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    decay_ratio = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(decay_ratio, 1.0)))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, device, eval_iters=30):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(model, optimizer, config, step, keep=3):
    STEP5_OUT.mkdir(parents=True, exist_ok=True)
    path = STEP5_OUT / f"checkpoint_step{step}.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": config.__dict__, "step": step}, path)
    ckpts = sorted(STEP5_OUT.glob("checkpoint_step*.pt"),
                   key=lambda p: int(p.stem.replace("checkpoint_step", "")))
    for old in ckpts[:-keep]:
        old.unlink()
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=24000)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--min-lr", type=float, default=4e-5)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--checkpoint-interval", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 64)
    print("SecGPTv2.5 — Pretraining (~100M from scratch)")
    print("=" * 64)

    config = GPTConfig()
    model = GPT(config).to(device)
    print(f"\n  Model: {model.num_parameters() / 1e6:.1f}M params on {device}")
    print(f"  Batch: {BATCH_SIZE} x {BLOCK_SIZE} = {BATCH_SIZE * BLOCK_SIZE:,} tokens/step")

    train_data = torch.load(STEP1_OUT / "train_data.bin", weights_only=True)
    val_data = torch.load(STEP1_OUT / "val_data.bin", weights_only=True)
    print(f"  Train: {len(train_data):,} | Val: {len(val_data):,} tokens")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda")
    start_step = 0
    if args.resume:
        ckpts = sorted(STEP5_OUT.glob("checkpoint_step*.pt"),
                       key=lambda p: int(p.stem.replace("checkpoint_step", "")))
        if ckpts:
            state = torch.load(ckpts[-1], map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_step = state["step"]
            print(f"  Resumed: {ckpts[-1].name} (step {start_step})")

    print(f"  Plan: {args.steps} steps, lr {args.lr} -> {args.min_lr} (cosine)")

    log = []
    t0 = time.time()
    model.train()
    for step in range(start_step, args.steps + 1):
        lr = get_lr(step, args.warmup, args.steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if step % args.eval_interval == 0 or step == args.steps:
            losses = estimate_loss(model, train_data, val_data, device)
            elapsed = time.time() - t0
            tok_s = (step - start_step) * BATCH_SIZE * BLOCK_SIZE / max(elapsed, 1)
            print(f"  step {step:>6} | lr {lr:.5f} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | {tok_s/1000:.1f}K tok/s | {elapsed/60:.1f}m",
                  flush=True)
            log.append({"step": step, "lr": round(lr, 6),
                        "train_loss": round(losses["train"], 4),
                        "val_loss": round(losses["val"], 4),
                        "elapsed_min": round(elapsed / 60, 1)})

        if step > 0 and step % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, config, step)

        if step == args.steps:
            break

        x, y = get_batch(train_data, device)
        with torch.amp.autocast("cuda"):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    total = time.time() - t0
    torch.save({"model": model.state_dict(), "config": config.__dict__,
                "step": args.steps}, STEP5_OUT / "checkpoint_final.pt")
    with open(STEP5_OUT / "train_log.json", "w") as f:
        json.dump({"config": config.__dict__, "steps": args.steps, "lr": args.lr,
                   "total_min": round(total / 60, 1), "params": model.num_parameters(),
                   "log": log}, f, indent=2)
    print(f"\n{'=' * 64}")
    print(f"  DONE in {total/3600:.2f}h — final val {log[-1]['val_loss']:.4f} "
          f"(random guess {math.log(config.vocab_size):.4f})")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
