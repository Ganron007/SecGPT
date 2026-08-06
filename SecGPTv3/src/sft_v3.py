"""
sft_v3.py — SecGPTv3 fairness run: GPT-2 Small SFT on the v3 dataset.

Same recipe as the Qwen line for a fair comparison: 500 steps, effective
batch 16 (4 x accum 4), block 512, Question/Answer format (GPT-2 has no
chat template). Checkpoints every 100 steps with full resume support.

Usage:
  python src/sft_v3.py                 # full run
  python src/sft_v3.py --steps 2       # smoke test
  python src/sft_v3.py --resume        # resume from latest checkpoint
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

from transformers import GPT2LMHeadModel, GPT2Tokenizer

ROOT = Path(__file__).resolve().parent.parent
SFT_DATA = ROOT.parent / "SecGPT-Prod" / "data" / "v3" / "sft_v3.jsonl"
OUTPUT = ROOT / "stage2_sft" / "output" / "model_v3"

MODEL_NAME = "openai-community/gpt2"
BLOCK_SIZE = 512
BATCH_SIZE = 4
ACCUM = 4
DEVICE = "cuda"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("SecGPTv3 — GPT-2 SFT on v3 data (fairness run)")
    print("=" * 60)

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    print(f"\n  Base: GPT-2 Small, {sum(p.numel() for p in model.parameters()):,} params")

    pairs = []
    with open(SFT_DATA, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    text = ""
    for p in pairs:
        text += f"Question: {p['instruction']}\nAnswer: {p['response']}\n\n"
    tokens = tokenizer.encode(text)
    data = torch.tensor(tokens, dtype=torch.long)
    print(f"  Data: {len(pairs):,} pairs, {len(data):,} tokens")

    split = int(len(data) * 0.95)
    train_data = data[:split]
    val_data = data[split:]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    start_step = 0
    ckpt_dir = OUTPUT / "checkpoints"
    if args.resume:
        ckpts = sorted(ckpt_dir.glob("step-*")) if ckpt_dir.exists() else []
        if ckpts:
            latest = ckpts[-1]
            state = torch.load(latest / "ckpt.pt", map_location=DEVICE)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_step = state["step"]
            print(f"  Resumed from {latest} (step {start_step})")

    def get_batch(split_data, bs=BATCH_SIZE):
        ix = torch.randint(len(split_data) - BLOCK_SIZE, (bs,))
        x = torch.stack([split_data[i:i + BLOCK_SIZE] for i in ix]).to(DEVICE)
        y = torch.stack([split_data[i + 1:i + BLOCK_SIZE + 1] for i in ix]).to(DEVICE)
        return x, y

    model.train()
    print(f"  Training: {args.steps} steps, lr={args.lr}, effective batch {BATCH_SIZE * ACCUM}")
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step, args.steps + 1):
        if step % 100 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                tl, vl = [], []
                for _ in range(8):
                    x, y = get_batch(train_data)
                    tl.append(model(input_ids=x, labels=y).loss.item())
                    x, y = get_batch(val_data)
                    vl.append(model(input_ids=x, labels=y).loss.item())
            print(f"  step {step:>4} | train {sum(tl)/8:.4f} | val {sum(vl)/8:.4f} | "
                  f"{time.time()-t0:.0f}s", flush=True)
            model.train()
            if step > 0:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                save_dir = ckpt_dir / f"step-{step}"
                save_dir.mkdir(exist_ok=True)
                torch.save({"model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "step": step}, save_dir / "ckpt.pt")
                for old in sorted(ckpt_dir.glob("step-*"))[:-2]:
                    (old / "ckpt.pt").unlink(missing_ok=True)
                    old.rmdir()

        if step == args.steps:
            break

        for _ in range(ACCUM):
            x, y = get_batch(train_data)
            loss = model(input_ids=x, labels=y).loss / ACCUM
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT)
    tokenizer.save_pretrained(OUTPUT)
    print(f"\n  Saved: {OUTPUT} ({(time.time()-t0)/60:.1f} min)")
    print("  Benchmark: python src/eval.py --model <SecGPTv3>/stage2_sft/output/model_v3 "
          "--no-lora --no-quant --prompt-format qa --name gpt2sftv3")


if __name__ == "__main__":
    main()
