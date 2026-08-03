"""
dpo_train.py — SecGPT-Prod Stage 2: DPO alignment (structured > verbose).

Policy    = base(4-bit) + SFT LoRA adapter (continued training).
Reference = separate base(4-bit) + same SFT adapter, frozen.
This keeps the KL anchor at the SFT model — NOT the raw base
(TRL's ref_model=None default would anchor to the wrong distribution).

Two 4-bit loads ≈ ~4.5 GB; total run fits in ~6 GB VRAM.

Usage:
  python src/dpo_train.py                      # full run: 4K pairs, 1 epoch
  python src/dpo_train.py --pairs 100 --max-steps 2   # pipeline validation
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
PAIRS_FILE = ROOT / "data" / "dpo_4k.jsonl"
SFT_LORA = ROOT / "stage1_sft" / "output" / "qwen_qlora" / "checkpoint-500"
OUTPUT_DIR = ROOT / "stage2_alignment" / "output" / "qwen_dpo"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def main():
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from trl import DPOConfig, DPOTrainer

    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=4000)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accum", type=int, default=8)
    args = parser.parse_args()

    print("=" * 60)
    print("SecGPT-Prod — DPO Alignment (structured > verbose)")
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

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)

    print(f"\n  Loading policy model: {MODEL_NAME} + SFT adapter (trainable)")
    policy = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
    policy = PeftModel.from_pretrained(policy, str(SFT_LORA), is_trainable=True)

    print("  Loading reference model: same weights, frozen")
    reference = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
    reference = PeftModel.from_pretrained(reference, str(SFT_LORA))
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = DPOConfig(
        output_dir=str(OUTPUT_DIR),
        beta=args.beta,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        max_length=512,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(model=policy, ref_model=reference, args=config,
                         train_dataset=dataset, processing_class=tokenizer)

    n_steps = args.max_steps if args.max_steps > 0 else \
        int(len(dataset) / (args.batch * args.accum) * args.epochs)
    print(f"\n  Config: beta={args.beta}, lr={args.lr}, effective batch "
          f"{args.batch * args.accum}, ~{n_steps} steps")
    print("  Starting DPO training...")
    t0 = time.time()
    trainer.train()
    total = time.time() - t0
    print(f"\n  Training complete in {total / 60:.1f} min")

    policy.save_pretrained(OUTPUT_DIR / "final")
    tokenizer.save_pretrained(OUTPUT_DIR / "final")
    log = {"model": MODEL_NAME, "sft_lora": str(SFT_LORA), "pairs": len(dataset),
           "beta": args.beta, "lr": args.lr, "epochs": args.epochs,
           "effective_batch": args.batch * args.accum,
           "train_time_min": round(total / 60, 1)}
    with open(OUTPUT_DIR / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR / 'final'}")
    print(f"\n{'=' * 60}")
    print("  DONE — benchmark with:")
    print(f"  python src/eval.py --lora {OUTPUT_DIR / 'final'} --name dpo")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
