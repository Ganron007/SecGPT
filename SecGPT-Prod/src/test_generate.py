"""test_generate.py — Quick generation test from any checkpoint."""
import sys, io, torch
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path

model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("stage2_sft/output/model")
print(f"Loading: {model_path}")
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path).to("cuda")
model.eval()

prompts = [
    "<|ttp|>\nQ: What is MITRE technique T1059?\nA:",
    "<|rule|>\nQ: Write a Sigma rule for suspicious PowerShell encoded command execution.\nA:",
    "<|ref|>\nQ: How can certutil.exe be abused for living-off-the-land attacks?\nA:",
]
for p in prompts:
    inputs = tokenizer(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    body = text[len(p):]
    print("\n" + "=" * 50)
    print("PROMPT:", p[:60])
    print("=" * 50)
    print(body[:400])
