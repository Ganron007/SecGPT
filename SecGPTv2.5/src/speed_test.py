"""
speed_test.py — measure v2.5 training throughput (tok/s) without training.

Runs timed forward+backward steps on real corpus batches with the same
AMP setup as train.py. Reports tok/s and peak VRAM per batch size.

Usage:
  python src/speed_test.py
"""

import io
import sys
import time
from pathlib import Path

import torch

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model import GPT, GPTConfig  # noqa: E402

BLOCK_SIZE = 512
STEP1_OUT = ROOT / "stage1_pre-training" / "step1_tokenizer" / "output"


def bench(batch_size, steps=15):
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42)
    config = GPTConfig()
    model = GPT(config).to("cuda")
    model.train()
    data = torch.load(STEP1_OUT / "train_data.bin", weights_only=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
    scaler = torch.amp.GradScaler("cuda")

    def get_batch():
        ix = torch.randint(len(data) - BLOCK_SIZE, (batch_size,))
        x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix]).to("cuda")
        y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix]).to("cuda")
        return x, y

    for _ in range(3):
        x, y = get_batch()
        with torch.amp.autocast("cuda"):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(steps):
        x, y = get_batch()
        with torch.amp.autocast("cuda"):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    dt = time.time() - t0
    toks = steps * batch_size * BLOCK_SIZE
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  batch {batch_size:>2}: {toks/dt:>8,.0f} tok/s | "
          f"{dt/steps*1000:>6.0f} ms/step | peak VRAM {vram:.2f} GB | loss {loss.item():.3f}")
    del model, optimizer, scaler, data
    torch.cuda.empty_cache()


def main():
    print("=" * 64)
    print("SecGPTv2.5 — SDPA speed test (98M, block 512, fp16 AMP)")
    print("=" * 64)
    for bs in (8, 12, 16):
        bench(bs)


if __name__ == "__main__":
    main()
