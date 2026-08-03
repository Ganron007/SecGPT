"""
sft_direct.py — Phase B corrected: SFT directly on base GPT-2 (no domain-adapt).

GPT-2 already knows English + has internet knowledge. We just teach it
to respond to security questions in structured format. No tags, no domain-adapt.

Usage:
  python src/sft_direct.py
  python src/sft_direct.py --generate
"""

import io
import json
import sys
import time
from pathlib import Path

import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

from transformers import GPT2LMHeadModel, GPT2Tokenizer

ROOT = Path(__file__).resolve().parent.parent
SFT_DATA = ROOT.parent / "SecGPTv2" / "stage2_sft" / "output" / "sft_data.jsonl"
OUTPUT = ROOT / "stage2_sft" / "output" / "model_direct"

MODEL_NAME = "openai-community/gpt2"
BLOCK_SIZE = 256
BATCH_SIZE = 4
DEVICE = "cuda"


def build_sft_text():
    pairs = []
    with open(SFT_DATA, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    text = ""
    for p in pairs:
        text += f"Question: {p['instruction']}\nAnswer: {p['response']}\n\n"
    return text, pairs


def generate(model, tokenizer, prompt, max_new_tokens=250, temperature=0.7):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.encode("\n\n")[0],
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    if "--generate" in sys.argv:
        model_path = OUTPUT
        if not model_path.exists():
            print("No model found. Run without --generate first.")
            return
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path).to(DEVICE)

        prompts = [
            "Question: What is MITRE technique T1059?\nAnswer:",
            "Question: Write a Sigma rule for suspicious PowerShell encoded command execution.\nAnswer:",
            "Question: How can certutil.exe be abused for living-off-the-land attacks?\nAnswer:",
            "Question: What is the Windows Prefetch artifact and why is it useful in forensics?\nAnswer:",
            "Question: Explain T1055 Process Injection and how adversaries use it.\nAnswer:",
        ]
        for p in prompts:
            text = generate(model, tokenizer, p)
            body = text[len(p):]
            print("\n" + "=" * 60)
            print(f"  {p[:70]}")
            print("=" * 60)
            print(body[:500])
        return

    print("=" * 60)
    print("Phase B — Direct SFT on GPT-2 Small (no domain-adapt)")
    print("=" * 60)

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n  Base model: GPT-2 Small, {params:,} params")

    sft_text, pairs = build_sft_text()
    tokens = tokenizer.encode(sft_text)
    data = torch.tensor(tokens, dtype=torch.long)
    print(f"  SFT data: {len(pairs)} pairs, {len(sft_text):,} chars, {len(data):,} tokens")

    split = int(len(data) * 0.9)
    train_data = data[:split]
    val_data = data[split:]

    steps = 300
    lr = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()

    print(f"  Training: {steps} steps, lr={lr}, batch={BATCH_SIZE}, block={BLOCK_SIZE}")
    print(f"\n{'Step':>6} | {'Train':>8} | {'Val':>8} | {'Time':>7}")
    print("-" * 45)

    def get_batch(split_data, bs=BATCH_SIZE):
        ix = torch.randint(len(split_data) - BLOCK_SIZE, (bs,))
        x = torch.stack([split_data[i:i + BLOCK_SIZE] for i in ix]).to(DEVICE)
        y = torch.stack([split_data[i + 1:i + BLOCK_SIZE + 1] for i in ix]).to(DEVICE)
        return x, y

    t0 = time.time()
    for step in range(steps + 1):
        if step % 100 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                tl, vl = [], []
                for _ in range(10):
                    x, y = get_batch(train_data)
                    tl.append(model(input_ids=x, labels=y).loss.item())
                    x, y = get_batch(val_data)
                    vl.append(model(input_ids=x, labels=y).loss.item())
            print(f"{step:>6} | {sum(tl)/10:>8.4f} | {sum(vl)/10:>8.4f} | {time.time()-t0:>6.1f}s")
            model.train()

        if step == steps:
            break

        x, y = get_batch(train_data)
        loss = model(input_ids=x, labels=y).loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT)
    tokenizer.save_pretrained(OUTPUT)
    print(f"\n  Saved: {OUTPUT} ({time.time()-t0:.0f}s)")
    print(f"  Test: python src/sft_direct.py --generate")


if __name__ == "__main__":
    main()
