# SecGPT-Prod — Usage Guide

A locally-running cybersecurity assistant. Qwen2.5-3B-Instruct + QLoRA, trained on 31K security Q&A pairs. Runs on RTX 4060 Laptop (8 GB VRAM).

---

## Requirements

Tested with: Python 3.14, PyTorch 2.11.0+cu128, transformers 5.8.1, peft 0.20.0, trl 1.9.2, datasets 5.0.1. Older versions within each major release should work but are untested (Python 3.11+ recommended).

Install:
```powershell
pip install -r requirements.txt
# or: pip install torch transformers peft bitsandbytes accelerate trl datasets
```

GPU: NVIDIA with 6+ GB VRAM (tested on RTX 4060 Laptop, 8 GB)

---

## Quick Start

```powershell
cd SecGPT-Prod
python src/quality_check.py
```

This loads the model and runs 14 benchmark prompts across all categories.

---

## Interactive Use (Python)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 1. Load base model in 4-bit
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization_config=bnb,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# 2. Load security LoRA adapters
LORA_PATH = "stage1_sft/output/qwen_qlora/checkpoint-500"  # run from SecGPT-Prod/
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# 3. Ask questions
def ask(question, max_tokens=300, temperature=0.7):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
```

---

## Example Prompts by Category

### MITRE ATT&CK / TTPs
```python
ask("What is MITRE technique T1059 and how do adversaries use it?")
ask("Explain T1055 Process Injection and how adversaries use it.")
ask("What is CVE-2021-44228 (Log4Shell) and how is it exploited?")
ask("Describe the kill chain model and its stages.")
ask("What is T1486 Data Encrypted for Impact (ransomware)?")
```

### Detection Rules (Sigma/YARA)
```python
ask("Write a Sigma detection rule for suspicious PowerShell encoded command execution.")
ask("Write a detection rule for WMI lateral movement.")
ask("Write a Sigma rule for LSASS memory access (credential dumping).")
ask("Write a YARA rule to detect Mimikatz in memory.")
ask("Write a detection rule for registry Run key persistence.")
```

### Living-off-the-Land / Tool Reference
```python
ask("How can certutil.exe be abused in a living-off-the-land attack?")
ask("How can mshta.exe be used to execute malicious code?")
ask("How can regsvr32.exe bypass application whitelisting?")
ask("How can rundll32.exe be abused for code execution?")
ask("How can wmic.exe be used for reconnaissance and lateral movement?")
```

### Forensics
```python
ask("What is the Windows Prefetch artifact and how is it useful in forensics?")
ask("What is the Amcache.hve forensic artifact and why is it useful?")
ask("What is the $MFT and how is it used in forensic investigations?")
ask("What is the ShimCache (AppCompatCache) and what does it prove?")
ask("What does the KAPE target SQLDatabases collect and why?")
```

### General Security Knowledge
```python
ask("What is the difference between a red team and a penetration test?")
ask("Explain the concept of defense in depth in cybersecurity.")
ask("What is threat hunting and how does it differ from incident response?")
ask("Describe the MITRE ATT&CK framework and its purpose.")
ask("Explain the concept of zero trust architecture.")
```

### Classification Tasks
```python
ask("Classify this SMS message as spam or ham: 'Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121'")
ask("Classify this network connection: 0,tcp,http,SF,29,45135,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,8,8,1.0,0.0,0.0,0.0,1.0,0.0,0.0,normal,21")
```

---

## Generation Parameters

| Parameter | Default | Effect |
|---|---|---|
| `max_new_tokens` | 300 | Response length limit |
| `temperature` | 0.7 | Creativity (lower = focused, higher = diverse) |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `do_sample` | True | Enable sampling (False = greedy/deterministic) |

**Recommended settings by task:**

| Task | Temperature | Max tokens |
|---|---|---|
| Detection rules | 0.5 | 400 |
| MITRE/CVE explanations | 0.7 | 300 |
| Tool abuse (LOLBAS) | 0.7 | 300 |
| General knowledge | 0.8 | 400 |
| Classification | 0.3 | 200 |

---

## Model Locations

| Component | Path | Size |
|---|---|---|
| Base model (Qwen2.5-3B) | `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/` | 5.76 GB |
| LoRA adapters (security) | `stage1_sft/output/qwen_qlora/checkpoint-500/adapter_model.safetensors` | 57 MB |
| Tokenizer | `stage1_sft/output/qwen_qlora/checkpoint-500/tokenizer.json` | 11 MB |
| Training dataset | `data/sft_32k.jsonl` | 15.8 MB |

**Note:** The base model downloads automatically from HuggingFace on first use (~5.76 GB). Subsequent loads use the local cache.

---

## VRAM Usage

| Phase | VRAM |
|---|---|
| Model loading (4-bit) | ~2.5 GB |
| LoRA adapter loading | +0.1 GB |
| Inference (300 tokens) | ~4.5 GB |
| Peak during generation | ~5.0 GB |
| **Total** | **~5 GB / 8 GB available** |

Headroom: 3 GB free. Can increase `max_new_tokens` to 512+ without OOM.

---

## Inference Speed

| Metric | Value |
|---|---|
| Model load time | ~15 seconds |
| First token latency | ~2-3 seconds |
| Generation speed | ~10-15 tokens/second |
| Full response (300 tokens) | ~20-30 seconds |

---

## Retraining / Fine-Tuning

```powershell
cd SecGPT-Prod

# Regenerate dataset (if corpus changed)
python src/build_sft_32k.py

# Retrain (full 1500 steps, ~3 hours)
python src/qlora_sft.py --steps 1500 --lr 2e-4

# Quick retrain (500 steps, ~1 hour)
python src/qlora_sft.py --steps 500 --lr 2e-4

# Quality check after training
python src/quality_check.py
```

### Training Options

| Flag | Default | Description |
|---|---|---|
| `--steps` | 1500 | Total training steps |
| `--lr` | 2e-4 | Learning rate |
| `--batch-size` | 4 | Per-device batch size |
| `--grad-accum` | 4 | Gradient accumulation (effective batch = 16) |
| `--lora-r` | 16 | LoRA rank (higher = more capacity) |
| `--lora-alpha` | 32 | LoRA scaling factor |

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `CUDA out of memory` | Reduce `max_new_tokens` or close other GPU apps |
| `ModuleNotFoundError: peft` | `pip install peft bitsandbytes accelerate` |
| Model download fails | Set `HF_TOKEN` env var, or download manually from huggingface.co |
| Slow generation | Normal (~10-15 tok/s on 4060). Use `max_new_tokens=200` for faster responses |
| Garbage output | Ensure LoRA path is correct. Check `adapter_model.safetensors` exists |
| `torch_dtype deprecated` warning | Cosmetic only, ignore |

---

## Limitations

- **Not a chatbot** — single-turn Q&A only (no conversation memory)
- **Can hallucinate** — verify critical facts against official sources (MITRE, NIST, vendor docs)
- **No tool use** — cannot execute commands, query APIs, or access files
- **Security-only** — performs poorly on non-security questions
- **No safety filter** — will answer offensive security questions (by design for red team study)
- **512 token context** — cannot process very long inputs

---

## Citing This Work

```
SecGPT-Prod: A locally-trained cybersecurity assistant
Base: Qwen2.5-3B-Instruct (Qwen Team, 2024)
Method: QLoRA SFT on 31K security Q&A pairs
Hardware: NVIDIA RTX 4060 Laptop (8 GB VRAM)
Dataset: DFIR-Nexus + CADRE KB + UCI SMS + NSL-KDD
```
