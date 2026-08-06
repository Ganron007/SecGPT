"""
dpo_train.py — SecGPTv2.5 Stage 3: DPO alignment.

Lessons applied (vs v2's collapsed GPT-2 run and v2's batch=1 DPO):
  beta 0.3, batched pairs (4/step), lr 1e-5.

Usage:
  python src/dpo_train.py
  python src/dpo_train.py --steps 10     # smoke test
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model import GPT, GPTConfig  # noqa: E402

SFT_CKPT = ROOT / "stage2_sft" / "output" / "checkpoint_sft.pt"
TOK_PATH = ROOT / "stage1_pre-training" / "step1_tokenizer" / "output" / "tokenizer.json"
DPO_DATA = ROOT / "stage3_alignment" / "output" / "dpo_data.jsonl"
OUTPUT = ROOT / "stage3_alignment" / "output"

BLOCK_SIZE = 512
BATCH_SIZE = 4


def get_log_probs(model, input_ids, target_ids):
    with torch.amp.autocast("cuda"):
        logits, _ = model(input_ids)
    log_probs = F.log_softmax(logits, dim=-1)
    tlp = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    mask = (target_ids != 0).float()
    return (tlp * mask).sum(-1) / mask.sum(-1).clamp(min=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 64)
    print("SecGPTv2.5 — Stage 3: DPO alignment")
    print("=" * 64)

    config = GPTConfig()
    policy = GPT(config).to(device)
    ckpt = torch.load(SFT_CKPT, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["model"])
    policy.train()
    reference = GPT(config).to(device)
    reference.load_state_dict(ckpt["model"])
    reference.eval()
    for p in reference.parameters():
        p.requires_grad = False
    print(f"\n  policy {policy.num_parameters()/1e6:.1f}M (trainable) | "
          f"reference frozen | beta={args.beta} | batch={BATCH_SIZE}")

    tokenizer = Tokenizer.from_file(str(TOK_PATH))
    pairs = []
    with open(DPO_DATA, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    print(f"  pairs: {len(pairs):,}")

    def encode_pair(pair):
        c = tokenizer.encode(pair["prompt"] + " " + pair["chosen"]).ids[:BLOCK_SIZE]
        r = tokenizer.encode(pair["prompt"] + " " + pair["rejected"]).ids[:BLOCK_SIZE]
        return c, r

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)
    t0 = time.time()
    idx = 0
    for step in range(args.steps + 1):
        if step % args.eval_interval == 0 or step == args.steps:
            policy.eval()
            losses, accs = [], []
            with torch.no_grad():
                for pair in pairs[:40]:
                    c, r = encode_pair(pair)
                    if len(c) < 10 or len(r) < 10:
                        continue
                    c_in = torch.tensor([c[:-1]], device=device)
                    c_tgt = torch.tensor([c[1:]], device=device)
                    r_in = torch.tensor([r[:-1]], device=device)
                    r_tgt = torch.tensor([r[1:]], device=device)
                    pi_c = get_log_probs(policy, c_in, c_tgt)
                    pi_r = get_log_probs(policy, r_in, r_tgt)
                    ref_c = get_log_probs(reference, c_in, c_tgt)
                    ref_r = get_log_probs(reference, r_in, r_tgt)
                    logits = args.beta * ((pi_c - pi_r) - (ref_c - ref_r))
                    losses.append(-F.logsigmoid(logits).mean().item())
                    accs.append((pi_c > pi_r).float().item())
            print(f"  step {step:>4} | loss {sum(losses)/len(losses):.4f} | "
                  f"acc {sum(accs)/len(accs):.1%} | {time.time()-t0:.0f}s", flush=True)
            policy.train()

        if step == args.steps:
            break

        batch_loss = 0
        for _ in range(BATCH_SIZE):
            pair = pairs[idx % len(pairs)]
            idx += 1
            c, r = encode_pair(pair)
            if len(c) < 10 or len(r) < 10:
                continue
            c_in = torch.tensor([c[:-1]], device=device)
            c_tgt = torch.tensor([c[1:]], device=device)
            r_in = torch.tensor([r[:-1]], device=device)
            r_tgt = torch.tensor([r[1:]], device=device)
            pi_c = get_log_probs(policy, c_in, c_tgt)
            pi_r = get_log_probs(policy, r_in, r_tgt)
            with torch.no_grad():
                ref_c = get_log_probs(reference, c_in, c_tgt)
                ref_r = get_log_probs(reference, r_in, r_tgt)
            logits = args.beta * ((pi_c - pi_r) - (ref_c - ref_r))
            batch_loss = batch_loss - F.logsigmoid(logits).mean()
        if isinstance(batch_loss, torch.Tensor):
            optimizer.zero_grad(set_to_none=True)
            (batch_loss / BATCH_SIZE).backward()
            optimizer.step()

    out_path = OUTPUT / "checkpoint_dpo.pt"
    torch.save({"model": policy.state_dict(), "config": config.__dict__,
                "step": args.steps, "beta": args.beta}, out_path)
    log = {"steps": args.steps, "lr": args.lr, "beta": args.beta,
           "total_min": round((time.time() - t0) / 60, 1), "pairs": len(pairs)}
    with open(OUTPUT / "dpo_train_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  DONE — saved {out_path.name}")


if __name__ == "__main__":
    main()
