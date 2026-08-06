"""
dpo_v3.py — SecGPTv3 fairness run: GPT-2 DPO on v3 preference pairs.

Policy    = GPT-2 SFT-v3 model (trainable).
Reference = frozen copy of the same SFT model.
Same DPO recipe as the Qwen line: beta=0.3, lr 5e-6, 1 epoch, eff. batch 16.

Usage:
  python src/dpo_v3.py                 # full run
  python src/dpo_v3.py --max-steps 2   # smoke test
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

ROOT = Path(__file__).resolve().parent.parent
PAIRS_FILE = ROOT.parent / "SecGPT-Prod" / "data" / "v3" / "dpo_v3_gpt2.jsonl"
SFT_MODEL = ROOT / "stage2_sft" / "output" / "model_v3"
OUTPUT_DIR = ROOT / "stage3_alignment" / "output" / "model_dpo_v3"
DEVICE = "cuda"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=4000)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accum", type=int, default=8)
    args = parser.parse_args()

    print("=" * 60)
    print("SecGPTv3 — GPT-2 DPO on v3 pairs (fairness run)")
    print("=" * 60)

    print(f"\n  Loading pairs: {PAIRS_FILE}")
    pairs = []
    with open(PAIRS_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                pairs.append({"prompt": o["prompt"], "chosen": o["chosen"],
                              "rejected": o["rejected"]})
    pairs = pairs[: args.pairs]
    dataset = Dataset.from_list(pairs)
    print(f"  Pairs: {len(dataset):,}")

    print(f"\n  Loading policy: {SFT_MODEL} (trainable)")
    policy = AutoModelForCausalLM.from_pretrained(str(SFT_MODEL)).to(DEVICE)
    print("  Loading reference: same weights, frozen")
    reference = AutoModelForCausalLM.from_pretrained(str(SFT_MODEL)).to(DEVICE)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(str(SFT_MODEL))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = DPOConfig(
        output_dir=str(OUTPUT_DIR),
        beta=args.beta,
        num_train_epochs=1.0,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        max_length=512,
        bf16=False,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(model=policy, ref_model=reference, args=config,
                         train_dataset=dataset, processing_class=tokenizer)

    n_steps = args.max_steps if args.max_steps > 0 else \
        int(len(dataset) / (args.batch * args.accum))
    print(f"\n  Config: beta={args.beta}, lr={args.lr}, effective batch "
          f"{args.batch * args.accum}, ~{n_steps} steps")
    print("  Starting DPO training...")
    t0 = time.time()
    trainer.train()
    total = time.time() - t0
    print(f"\n  Training complete in {total / 60:.1f} min")

    policy.save_pretrained(OUTPUT_DIR / "final")
    tokenizer.save_pretrained(OUTPUT_DIR / "final")
    log = {"model": "gpt2-sft-v3", "pairs": len(dataset), "beta": args.beta,
           "lr": args.lr, "effective_batch": args.batch * args.accum,
           "train_time_min": round(total / 60, 1)}
    with open(OUTPUT_DIR / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR / 'final'}")
    print(f"\n{'=' * 60}")
    print("  DONE — benchmark with:")
    print(f"  python src/eval.py --model {OUTPUT_DIR / 'final'} --no-lora "
          f"--no-quant --prompt-format qa --name gpt2dpov3")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
