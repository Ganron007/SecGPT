"""
dpo_train.py — Phase A Stage 3: DPO alignment for SecGPT v2.

Direct Preference Optimization: teaches the model to PREFER structured,
complete answers over degraded ones. No reward model, no RL loop —
just a supervised loss on preference pairs.

DPO loss = -log(sigmoid(beta * (log_pi(chosen) - log_pi(rejected) - log_ref(chosen) + log_ref(rejected))))

Usage:
  python src/dpo_train.py
  python src/dpo_train.py --steps 500 --beta 0.1
"""

import argparse
import copy
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
from src.model import GPT, GPTConfig

SFT_CKPT = ROOT / "stage2_sft" / "output" / "checkpoint_sft.pt"
TOK_PATH = ROOT / "stage1_pre-training" / "step1_tokenizer" / "output" / "tokenizer.json"
DPO_DATA = ROOT / "stage3_alignment" / "output" / "dpo_data.jsonl"
OUTPUT = ROOT / "stage3_alignment" / "output"

BLOCK_SIZE = 256
BATCH_SIZE = 4


def get_log_probs(model, input_ids, target_ids, device):
    with torch.amp.autocast("cuda"):
        logits, _ = model(input_ids)
    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    mask = (target_ids != 0).float()
    return (target_log_probs * mask).sum(-1) / mask.sum(-1).clamp(min=1)


def main():
    parser = argparse.ArgumentParser(description="SecGPT v2 DPO Alignment")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature (KL penalty strength)")
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Phase A — Stage 3: DPO Alignment")
    print("=" * 60)

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

    print(f"\n  Policy model: {policy.num_parameters():,} params (trainable)")
    print(f"  Reference model: {reference.num_parameters():,} params (frozen)")
    print(f"  DPO beta: {args.beta}")
    print(f"  Training: {args.steps} steps, lr={args.lr}")

    tokenizer = Tokenizer.from_file(str(TOK_PATH))

    pairs = []
    with open(DPO_DATA, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    print(f"  Preference pairs: {len(pairs)}")

    def encode_pair(pair):
        prompt = pair["prompt"]
        chosen_text = prompt + " " + pair["chosen"]
        rejected_text = prompt + " " + pair["rejected"]
        chosen_ids = tokenizer.encode(chosen_text).ids[:BLOCK_SIZE]
        rejected_ids = tokenizer.encode(rejected_text).ids[:BLOCK_SIZE]
        return chosen_ids, rejected_ids

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"\n{'Step':>6} | {'DPO Loss':>8} | {'Acc':>6} | {'Time':>7}")
    print("-" * 40)

    t0 = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_interval == 0 or step == args.steps:
            policy.eval()
            eval_losses = []
            eval_accs = []
            with torch.no_grad():
                for pair in pairs[:50]:
                    chosen_ids, rejected_ids = encode_pair(pair)
                    if len(chosen_ids) < 10 or len(rejected_ids) < 10:
                        continue
                    c_in = torch.tensor([chosen_ids[:-1]], device=device)
                    c_tgt = torch.tensor([chosen_ids[1:]], device=device)
                    r_in = torch.tensor([rejected_ids[:-1]], device=device)
                    r_tgt = torch.tensor([rejected_ids[1:]], device=device)

                    pi_c = get_log_probs(policy, c_in, c_tgt, device)
                    pi_r = get_log_probs(policy, r_in, r_tgt, device)
                    ref_c = get_log_probs(reference, c_in, c_tgt, device)
                    ref_r = get_log_probs(reference, r_in, r_tgt, device)

                    logits = args.beta * ((pi_c - pi_r) - (ref_c - ref_r))
                    loss = -F.logsigmoid(logits).mean()
                    eval_losses.append(loss.item())
                    eval_accs.append((pi_c > pi_r).float().item())

            elapsed = time.time() - t0
            avg_loss = sum(eval_losses) / len(eval_losses) if eval_losses else 0
            avg_acc = sum(eval_accs) / len(eval_accs) if eval_accs else 0
            print(f"{step:>6} | {avg_loss:>8.4f} | {avg_acc:>5.1%} | {elapsed:>6.1f}s")
            policy.train()

        if step == args.steps:
            break

        pair = pairs[step % len(pairs)]
        chosen_ids, rejected_ids = encode_pair(pair)
        if len(chosen_ids) < 10 or len(rejected_ids) < 10:
            continue

        c_in = torch.tensor([chosen_ids[:-1]], device=device)
        c_tgt = torch.tensor([chosen_ids[1:]], device=device)
        r_in = torch.tensor([rejected_ids[:-1]], device=device)
        r_tgt = torch.tensor([rejected_ids[1:]], device=device)

        pi_c = get_log_probs(policy, c_in, c_tgt, device)
        pi_r = get_log_probs(policy, r_in, r_tgt, device)
        with torch.no_grad():
            ref_c = get_log_probs(reference, c_in, c_tgt, device)
            ref_r = get_log_probs(reference, r_in, r_tgt, device)

        logits = args.beta * ((pi_c - pi_r) - (ref_c - ref_r))
        loss = -F.logsigmoid(logits).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total = time.time() - t0
    out_path = OUTPUT / "checkpoint_dpo.pt"
    torch.save({"model": policy.state_dict(), "config": config.__dict__, "step": args.steps, "beta": args.beta}, out_path)
    print(f"\n  DPO complete in {total:.1f}s")
    print(f"  Checkpoint: {out_path.name} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    log = {"steps": args.steps, "lr": args.lr, "beta": args.beta, "total_time_s": round(total, 1), "pairs": len(pairs), "base": "stage2_sft checkpoint"}
    with open(OUTPUT / "dpo_train_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  DONE — model now prefers structured answers over degraded ones")
    print(f"  Test: python src/generate.py --checkpoint stage3_alignment/output/checkpoint_dpo.pt --prompt \"<|ttp|>\\nQ: What is T1059?\\nA:\"")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
