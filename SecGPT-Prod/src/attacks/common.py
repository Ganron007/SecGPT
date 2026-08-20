"""
common.py — Phase 2 attack harness: shared model loading + generation.

Supports all three SecGPT model lines:
  qwen  — SecGPT-Prod (Qwen2.5-3B + LoRA adapter, 4-bit)
  gpt2  — SecGPTv3 (GPT-2 fine-tune, plain HF)
  v25   — SecGPTv2.5 (custom GPT, own tokenizer)

Usage:
  from attacks.common import load_model, generate, SCORERS
"""

import io
import json
import sys
from pathlib import Path

import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent.parent  # SecGPT-Prod
PROD = ROOT  # SecGPT-Prod
V3 = ROOT.parent / "SecGPTv3"
V25 = ROOT.parent / "SecGPTv2.5"

QWEN_LORA = PROD / "stage2_alignment" / "output" / "qwen_dpo_v3" / "final"
GPT2_MODEL = V3 / "stage2_sft" / "output" / "model_v3"
V25_CKPT = V25 / "stage3_alignment" / "output" / "checkpoint_dpo.pt"
V25_TOK = V25 / "stage1_pre-training" / "step1_tokenizer" / "output" / "tokenizer.json"

MODELS = {"qwen": "qwen", "gpt2": "gpt2", "v25": "v25"}


def load_model(name):
    import torch
    name = name.lower()
    assert name in MODELS, f"unknown model {name}; use {list(MODELS)}"
    if name == "qwen":
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        base = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct", quantization_config=bnb,
            device_map="auto", torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, str(QWEN_LORA))
        model.eval()
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        return {"model": model, "tokenizer": tok, "kind": "qwen"}
    if name == "gpt2":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(str(GPT2_MODEL)).to("cuda")
        model.eval()
        tok = AutoTokenizer.from_pretrained(str(GPT2_MODEL))
        tok.pad_token = tok.eos_token
        return {"model": model, "tokenizer": tok, "kind": "gpt2"}
    if name == "v25":
        import sys as _sys
        _sys.path.insert(0, str(V25))
        from src.model import GPT, GPTConfig  # noqa: F401
        from tokenizers import Tokenizer
        ckpt = torch.load(V25_CKPT, map_location="cuda", weights_only=False)
        config = GPTConfig(**ckpt["config"])
        model = GPT(config).to("cuda")
        model.load_state_dict(ckpt["model"])
        model.eval()
        tok = Tokenizer.from_file(str(V25_TOK))
        return {"model": model, "tokenizer": tok, "kind": "v25"}


def make_prompt(kind, text, system=None):
    if kind == "qwen":
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": text})
        return messages
    if kind == "gpt2":
        prefix = f"System: {system}\n\n" if system else ""
        return f"{prefix}Question: {text}\nAnswer:"
    return None  # v25 handled by make_prompt_v25


def make_prompt_v25(text, tag="kb", system=None):
    prefix = f"<|sys|>\n{system}\n\n" if system else ""
    return f"{prefix}<|{tag}|>\nQ: {text}\nA:"


TAG_FOR = {"ttp": "ttp", "rule": "rule", "ref": "ref", "kb": "kb",
           "classification": "kb"}


@torch.no_grad()
def generate(handle, prompt, max_new=300, system=None):
    kind = handle["kind"]
    model, tok = handle["model"], handle["tokenizer"]
    if kind == "qwen":
        msgs = prompt if isinstance(prompt, list) else make_prompt(kind, prompt, system)
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    if kind == "gpt2":
        p = prompt if prompt.startswith("Question:") else make_prompt(kind, prompt, system)
        ids = tok(p, return_tensors="pt").to("cuda")
        nl2 = tok.encode("\n\n", add_special_tokens=False)
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id,
                             eos_token_id=[tok.eos_token_id] + (nl2[:1] if nl2 else []))
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    # v25
    if isinstance(prompt, list):
        prompt = prompt[-1]["content"] if prompt else ""
    p = make_prompt_v25(prompt, system=system)
    ids = tok.encode(p).ids
    idx = torch.tensor([ids], dtype=torch.long, device="cuda")
    nl = tok.encode("\n").ids[0]
    for _ in range(max_new):
        cond = idx[:, -model.config.block_size:]
        logits, _ = model(cond)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat((idx, nxt), dim=1)
        if idx[0, -1].item() == nl and idx[0, -2].item() == nl:
            break
    ans = tok.decode(idx[0, len(ids):].tolist())
    for stop in ("\n\nQ:", "\n\n<|", "\nQ:"):
        if stop in ans:
            ans = ans[:ans.index(stop)]
    return ans.strip()


def save_results(name, results, meta=None):
    import time as _t
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _t.strftime("%Y%m%d_%H%M")
    path = out_dir / f"attack_{name}_{ts}.json"
    payload = {"attack": name, "timestamp": ts, "meta": meta or {}, "results": results}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path
