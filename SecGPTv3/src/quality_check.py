"""
quality_check.py — SecGPTv3: Generate from trained QLoRA model and evaluate quality.
"""
import io
import json
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
LORA_PATH = ROOT / "stage1_sft" / "output" / "phi3_qlora" / "checkpoint-500"
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEVICE = "cuda"

print("Loading base model (4-bit)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16)
print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(model, str(LORA_PATH))
model.eval()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def generate(prompt, max_tokens=300, temp=0.7):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temp, do_sample=True, top_p=0.9)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

prompts = [
    ("ttp", "What is MITRE technique T1059 and how do adversaries use it?"),
    ("ttp", "Explain T1055 Process Injection and how adversaries use it."),
    ("ttp", "What is CVE-2021-44228 (Log4Shell) and how is it exploited?"),
    ("rule", "Write a Sigma detection rule for suspicious PowerShell encoded command execution."),
    ("rule", "Write a detection rule for WMI lateral movement."),
    ("rule", "Write a Sigma rule for LSASS memory access (credential dumping)."),
    ("ref", "How can certutil.exe be abused in a living-off-the-land attack?"),
    ("ref", "How can mshta.exe be used to execute malicious code?"),
    ("ref", "What is the Windows Prefetch artifact and how is it useful in forensics?"),
    ("kb", "What is the difference between a red team and a penetration test?"),
    ("kb", "Explain the concept of defense in depth in cybersecurity."),
    ("kb", "What is threat hunting and how does it differ from incident response?"),
    ("classification", "Classify this SMS message as spam or ham: 'Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121'"),
    ("classification", "Classify this network connection: 0,tcp,http,SF,29,45135,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,8,8,1.0,0.0,0.0,0.0,1.0,0.0,0.0,normal,21"),
]

print("\n" + "=" * 70)
print("SecGPTv3 Quality Check — Qwen2.5-3B + QLoRA (31K security Q&A)")
print("=" * 70)

results = []
for cat, prompt in prompts:
    print(f"\n{'─' * 70}")
    print(f"[{cat.upper()}] {prompt}")
    print(f"{'─' * 70}")
    response = generate(prompt)
    print(response[:500])
    results.append({"category": cat, "prompt": prompt, "response": response})

out_path = ROOT / "stage1_sft" / "output" / "quality_check_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n\nSaved: {out_path}")
