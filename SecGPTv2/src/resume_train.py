"""
resume_train.py — Resume SecGPT v2 training from checkpoint_step10000.pt.
"""

import io
import math
import sys
import time
from pathlib import Path

import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model import GPT, GPTConfig

device = "cuda"
config = GPTConfig()
model = GPT(config).to(device)

ckpt_path = ROOT / "stage1_pre-training" / "step5_training" / "output" / "checkpoint_step10000.pt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
start_step = ckpt["step"]
print(f"Resumed from step {start_step}")

train_data = torch.load(ROOT / "stage1_pre-training" / "step1_tokenizer" / "output" / "train_data.bin", weights_only=True)
val_data = torch.load(ROOT / "stage1_pre-training" / "step1_tokenizer" / "output" / "val_data.bin", weights_only=True)
print(f"Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")

BLOCK_SIZE = 256
BATCH_SIZE = 32
END_STEP = 20000
MAX_LR = 6e-4
MIN_LR = 6e-5
WARMUP = 500


def get_batch(data):
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)


def get_lr(step):
    if step < WARMUP:
        return MAX_LR * step / WARMUP
    decay = (step - WARMUP) / (END_STEP - WARMUP)
    return MIN_LR + 0.5 * (1.0 + math.cos(math.pi * decay)) * (MAX_LR - MIN_LR)


optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.1)
scaler = torch.amp.GradScaler("cuda")

print(f"\nTraining {start_step} -> {END_STEP}")
print(f"{'Step':>6} | {'Train':>8} | {'Val':>8} | {'Time':>7}")
print("-" * 45)

t0 = time.time()
model.train()

for step in range(start_step, END_STEP + 1):
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    if step % 1000 == 0 or step == END_STEP:
        model.eval()
        with torch.no_grad():
            train_losses = [model(*get_batch(train_data))[1].item() for _ in range(30)]
            val_losses = [model(*get_batch(val_data))[1].item() for _ in range(30)]
        elapsed = time.time() - t0
        print(f"{step:>6} | {sum(train_losses)/30:>8.4f} | {sum(val_losses)/30:>8.4f} | {elapsed:>6.1f}s")
        model.train()

    if step == END_STEP:
        break

    x, y = get_batch(train_data)
    with torch.amp.autocast("cuda"):
        _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

total = time.time() - t0
out_path = ROOT / "stage1_pre-training" / "step5_training" / "output" / "checkpoint_final.pt"
torch.save({"model": model.state_dict(), "config": config.__dict__, "step": END_STEP}, out_path)
print(f"\nDone in {total:.0f}s ({total/60:.1f} min)")
print(f"Checkpoint: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
