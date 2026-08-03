"""
qlora_sft.py — SecGPT-Prod: QLoRA SFT on Qwen2.5-3B-Instruct with 31K security Q&A pairs.

4-bit quantized base + LoRA adapters. Fits in ~5 GB VRAM.

Usage:
  python src/qlora_sft.py
  python src/qlora_sft.py --steps 2000 --lr 2e-4
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sft_32k.jsonl"
OUTPUT_DIR = ROOT / "stage1_sft" / "output" / "qwen_qlora"

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEQ_LEN = 512


def format_example(example):
    text = f"<|user|>\n{example['instruction']}<|end|>\n<|assistant|>\n{example['response']}<|end|>"
    return {"text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    print("=" * 60)
    print("SecGPT-Prod — Qwen2.5-3B QLoRA SFT")
    print("=" * 60)

    print(f"\n  Loading dataset: {DATA_PATH}")
    pairs = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    dataset = Dataset.from_list(pairs)
    dataset = dataset.map(format_example)
    print(f"  Dataset: {len(dataset):,} examples")

    print(f"\n  Loading {MODEL_NAME} (4-bit quantized)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    params = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model loaded: {params:,} total params")

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA adapters: {trainable:,} trainable params ({trainable/params*100:.2f}% of total)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        optim="paged_adamw_8bit",
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"\n  Training config:")
    print(f"    Steps: {args.steps}")
    print(f"    Effective batch: {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"    LR: {args.lr} (cosine, 100 warmup)")
    print(f"    LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"    Max seq len: {MAX_SEQ_LEN}")
    print(f"    Optimizer: paged_adamw_8bit")
    print(f"\n  Starting training...")

    t0 = time.time()
    trainer.train()
    total = time.time() - t0

    print(f"\n  Training complete in {total:.0f}s ({total/60:.1f} min)")

    model.save_pretrained(OUTPUT_DIR / "final")
    tokenizer.save_pretrained(OUTPUT_DIR / "final")
    print(f"  Saved: {OUTPUT_DIR / 'final'}")

    log = {
        "model": MODEL_NAME,
        "params_total": params,
        "params_trainable": trainable,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "steps": args.steps,
        "lr": args.lr,
        "batch_effective": args.batch_size * args.grad_accum,
        "dataset_size": len(dataset),
        "total_time_s": round(total, 1),
        "max_seq_len": MAX_SEQ_LEN,
    }
    with open(OUTPUT_DIR / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  DONE — Qwen2.5-3B + LoRA trained on {len(dataset):,} security Q&A pairs")
    print(f"  Test: python src/quality_check.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
