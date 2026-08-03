"""
pipeline.py — Phase B: Full LLM pipeline on pretrained GPT-2 Small (124M).

The company approach: take an existing fluent model, adapt it to security domain.
  Stage 1: Domain-adaptive pretraining on security corpus
  Stage 2: SFT on Q&A pairs
  Stage 3: DPO alignment

Usage:
  python src/pipeline.py                    # run all 3 stages
  python src/pipeline.py --stage 1          # domain-adapt only
  python src/pipeline.py --stage 2          # SFT only (needs stage 1 done)
  python src/pipeline.py --stage 3          # DPO only (needs stage 2 done)
  python src/pipeline.py --generate         # test generation from final model
"""

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT.parent / "SecGPTv2" / "stage1_pre-training" / "step0_corpus" / "output" / "corpus.txt"
SFT_CORPUS = ROOT.parent / "SecGPTv2" / "stage2_sft" / "output" / "sft_corpus.txt"
DPO_DATA = ROOT.parent / "SecGPTv2" / "stage3_alignment" / "output" / "dpo_data.jsonl"

STAGE1_OUT = ROOT / "stage1_domain-adapt" / "output"
STAGE2_OUT = ROOT / "stage2_sft" / "output"
STAGE3_OUT = ROOT / "stage3_alignment" / "output"

MODEL_NAME = "openai-community/gpt2"
BLOCK_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_base_model():
    print(f"  Loading {MODEL_NAME}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"  GPT-2 Small: {params:,} params ({params/1e6:.0f}M) on {DEVICE}")
    print(f"  Vocab: {tokenizer.vocab_size}, Context: {model.config.n_positions}")
    return model, tokenizer


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


def eval_loss(model, data, block_size, batch_size, iters=10):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(iters):
            x, y = get_batch(data, block_size, batch_size)
            out = model(input_ids=x, labels=y)
            losses.append(out.loss.item())
    model.train()
    return sum(losses) / len(losses)


def stage1_domain_adapt(model, tokenizer, steps=2000, lr=2e-5, batch_size=8):
    print("\n" + "=" * 60)
    print("Stage 1: Domain-Adaptive Pretraining")
    print("=" * 60)

    corpus_text = CORPUS.read_text(encoding="utf-8")
    print(f"  Corpus: {len(corpus_text)/1024/1024:.1f} MB")
    tokens = tokenizer.encode(corpus_text)
    data = torch.tensor(tokens, dtype=torch.long)
    print(f"  Encoded: {len(data):,} tokens (GPT-2 BPE, {len(corpus_text)/len(data):.2f} chars/token)")

    split = int(len(data) * 0.95)
    train_data = data[:split]
    val_data = data[split:]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()

    print(f"  Training: {steps} steps, lr={lr}, batch={batch_size}, block={BLOCK_SIZE}")
    print(f"\n{'Step':>6} | {'Train':>8} | {'Val':>8} | {'Time':>7}")
    print("-" * 45)

    t0 = time.time()
    for step in range(steps + 1):
        if step % 500 == 0 or step == steps:
            tl = eval_loss(model, train_data, BLOCK_SIZE, batch_size)
            vl = eval_loss(model, val_data, BLOCK_SIZE, batch_size)
            print(f"{step:>6} | {tl:>8.4f} | {vl:>8.4f} | {time.time()-t0:>6.1f}s")

        if step == steps:
            break

        x, y = get_batch(train_data, BLOCK_SIZE, batch_size)
        outputs = model(input_ids=x, labels=y)
        loss = outputs.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    STAGE1_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(STAGE1_OUT / "model")
    tokenizer.save_pretrained(STAGE1_OUT / "model")
    print(f"\n  Saved: stage1_domain-adapt/output/model/ ({time.time()-t0:.0f}s total)")
    return model, tokenizer


def stage2_sft(model, tokenizer, steps=1000, lr=1e-5, batch_size=4):
    print("\n" + "=" * 60)
    print("Stage 2: Supervised Fine-Tuning")
    print("=" * 60)

    sft_text = SFT_CORPUS.read_text(encoding="utf-8")
    tokens = tokenizer.encode(sft_text)
    data = torch.tensor(tokens, dtype=torch.long)
    print(f"  SFT corpus: {len(sft_text):,} chars → {len(data):,} tokens")

    split = int(len(data) * 0.9)
    train_data = data[:split]
    val_data = data[split:]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()

    print(f"  Training: {steps} steps, lr={lr}, batch={batch_size}")
    print(f"\n{'Step':>6} | {'Train':>8} | {'Val':>8} | {'Time':>7}")
    print("-" * 45)

    t0 = time.time()
    for step in range(steps + 1):
        if step % 200 == 0 or step == steps:
            tl = eval_loss(model, train_data, BLOCK_SIZE, batch_size)
            vl = eval_loss(model, val_data, BLOCK_SIZE, batch_size)
            print(f"{step:>6} | {tl:>8.4f} | {vl:>8.4f} | {time.time()-t0:>6.1f}s")

        if step == steps:
            break

        x, y = get_batch(train_data, BLOCK_SIZE, batch_size)
        outputs = model(input_ids=x, labels=y)
        loss = outputs.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    STAGE2_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(STAGE2_OUT / "model")
    tokenizer.save_pretrained(STAGE2_OUT / "model")
    print(f"\n  Saved: stage2_sft/output/model/ ({time.time()-t0:.0f}s total)")
    return model, tokenizer


def stage3_dpo(model, tokenizer, steps=300, lr=5e-6, beta=0.1, batch_size=1):
    print("\n" + "=" * 60)
    print("Stage 3: DPO Alignment")
    print("=" * 60)

    pairs = []
    with open(DPO_DATA, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    print(f"  Preference pairs: {len(pairs)}")

    import copy
    reference = copy.deepcopy(model)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()

    def get_log_probs(m, ids):
        inp = ids[:, :-1]
        tgt = ids[:, 1:]
        with torch.amp.autocast("cuda"):
            logits = m(inp).logits
        lp = F.log_softmax(logits, dim=-1)
        tgt_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        mask = (tgt != tokenizer.eos_token_id).float()
        return (tgt_lp * mask).sum(-1) / mask.sum(-1).clamp(min=1)

    print(f"  Training: {steps} steps, lr={lr}, beta={beta}")
    print(f"\n{'Step':>6} | {'DPO Loss':>8} | {'Acc':>6} | {'Time':>7}")
    print("-" * 40)

    t0 = time.time()
    for step in range(steps + 1):
        if step % 100 == 0 or step == steps:
            model.eval()
            losses, accs = [], []
            with torch.no_grad():
                for pair in pairs[:30]:
                    prompt = pair["prompt"]
                    c_ids = torch.tensor([tokenizer.encode(prompt + " " + pair["chosen"])[:BLOCK_SIZE]], device=DEVICE)
                    r_ids = torch.tensor([tokenizer.encode(prompt + " " + pair["rejected"])[:BLOCK_SIZE]], device=DEVICE)
                    if c_ids.shape[1] < 10 or r_ids.shape[1] < 10:
                        continue
                    pi_c = get_log_probs(model, c_ids)
                    pi_r = get_log_probs(model, r_ids)
                    ref_c = get_log_probs(reference, c_ids)
                    ref_r = get_log_probs(reference, r_ids)
                    logit = beta * ((pi_c - pi_r) - (ref_c - ref_r))
                    losses.append(-F.logsigmoid(logit).item())
                    accs.append((pi_c > pi_r).float().item())
            avg_l = sum(losses) / len(losses) if losses else 0
            avg_a = sum(accs) / len(accs) if accs else 0
            print(f"{step:>6} | {avg_l:>8.4f} | {avg_a:>5.1%} | {time.time()-t0:>6.1f}s")
            model.train()

        if step == steps:
            break

        pair = pairs[step % len(pairs)]
        prompt = pair["prompt"]
        c_ids = torch.tensor([tokenizer.encode(prompt + " " + pair["chosen"])[:BLOCK_SIZE]], device=DEVICE)
        r_ids = torch.tensor([tokenizer.encode(prompt + " " + pair["rejected"])[:BLOCK_SIZE]], device=DEVICE)
        if c_ids.shape[1] < 10 or r_ids.shape[1] < 10:
            continue

        pi_c = get_log_probs(model, c_ids)
        pi_r = get_log_probs(model, r_ids)
        with torch.no_grad():
            ref_c = get_log_probs(reference, c_ids)
            ref_r = get_log_probs(reference, r_ids)

        logit = beta * ((pi_c - pi_r) - (ref_c - ref_r))
        loss = -F.logsigmoid(logit).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    STAGE3_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(STAGE3_OUT / "model")
    tokenizer.save_pretrained(STAGE3_OUT / "model")
    print(f"\n  Saved: stage3_alignment/output/model/ ({time.time()-t0:.0f}s total)")
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.7):
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
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="SecGPTv3 Pipeline")
    parser.add_argument("--stage", type=int, default=0, help="Run specific stage (1/2/3), 0=all")
    parser.add_argument("--generate", action="store_true", help="Test generation from final model")
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()

    if args.generate:
        model_path = STAGE3_OUT / "model"
        if not model_path.exists():
            model_path = STAGE2_OUT / "model"
        if not model_path.exists():
            model_path = STAGE1_OUT / "model"
        print(f"  Loading from {model_path}")
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path).to(DEVICE)
        prompts = [
            "<|ttp|>\nQ: What is MITRE technique T1059?\nA:",
            "<|rule|>\nQ: Write a Sigma rule for suspicious PowerShell execution\nA:",
            "<|ref|>\nQ: How can certutil.exe be abused for living-off-the-land?\nA:",
        ]
        if args.prompt:
            prompts = [args.prompt.replace("\\n", "\n")]
        for p in prompts:
            print(f"\n{'='*50}\n  {p[:60]}...\n{'='*50}")
            text = generate(model, tokenizer, p)
            body = text[len(p):]
            print(f"\n{body[:400]}")
        return

    model, tokenizer = load_base_model()

    if args.stage == 0 or args.stage == 1:
        model, tokenizer = stage1_domain_adapt(model, tokenizer)
    if args.stage == 0 or args.stage == 2:
        if args.stage == 2:
            model = GPT2LMHeadModel.from_pretrained(STAGE1_OUT / "model").to(DEVICE)
            tokenizer = GPT2Tokenizer.from_pretrained(STAGE1_OUT / "model")
        model, tokenizer = stage2_sft(model, tokenizer)
    if args.stage == 0 or args.stage == 3:
        if args.stage == 3:
            model = GPT2LMHeadModel.from_pretrained(STAGE2_OUT / "model").to(DEVICE)
            tokenizer = GPT2Tokenizer.from_pretrained(STAGE2_OUT / "model")
        model, tokenizer = stage3_dpo(model, tokenizer)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("  Test: python src/pipeline.py --generate")
    print("=" * 60)


if __name__ == "__main__":
    main()
