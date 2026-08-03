---
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-3B-Instruct
- lora
- sft
- transformers
- trl
- cybersecurity
---

# SecGPT-Prod — Qwen2.5-3B Cybersecurity LoRA (checkpoint-500)

QLoRA adapter that turns Qwen2.5-3B-Instruct into a single-turn cybersecurity
Q&A assistant: MITRE ATT&CK/CAPEC/KEV explanations, Sigma-style detection
rules, LOLBAS/GTFOBins abuse references, DFIR artifact explanations, and
SMS/network-traffic classification.

## Model Details

- **Developed by:** SecGPT project (HTB AI Red Teamer certification study)
- **Model type:** LoRA adapter (r=16, alpha=32, all q/k/v/o/gate/up/down proj) over a 4-bit NF4 quantized base
- **Language(s) (NLP):** English
- **License:** Educational/research use — see repo LICENSE; base model under the Qwen License
- **Finetuned from model:** Qwen/Qwen2.5-3B-Instruct

### Model Sources

- **Repository:** https://github.com/ (SecGPT — see `SecGPT-Prod/doc.md` and `USAGE.md`)

## Uses

### Direct Use

Single-turn security Q&A via the Qwen chat template. Load base model in
4-bit, then apply this adapter (see `SecGPT-Prod/USAGE.md` for full code):

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "checkpoint-500")
```

### Out-of-Scope Use

- General-purpose chat (degraded outside the security domain, by design)
- Source of truth for critical facts — can hallucinate technique IDs; verify against MITRE/NIST/vendor docs
- Unauthorized offensive activity — offensive-security content is for authorized research/education only

## Bias, Risks, and Limitations

- No safety alignment: answers offensive as well as defensive security questions (intentional, for red team study)
- 512-token training context; no multi-turn memory; no tool use
- Trained on a corpus that includes commercial course material — do not redistribute generated content at scale

## Training Details

### Training Data

31,111 instruction/response pairs generated from DFIR-Nexus structured
sources (MITRE ATT&CK, Sigma, Elastic, Atomic, CAPEC, CISA KEV, Splunk,
Hayabusa, Chainsaw, LOLBAS, GTFOBins, forensic artifacts, KAPE,
Velociraptor, HijackLibs, LOLDrivers, MBC, D3FEND), CADRE KB chunks,
UCI SMS Spam, and NSL-KDD. Built by `src/build_sft_32k.py` (seed 42).

### Training Procedure

QLoRA SFT via TRL `SFTTrainer`: NF4 4-bit + double quantization, bf16
compute, paged_adamw_8bit, cosine LR 2e-4 (100 warmup), effective batch 16,
max seq len 512. Stopped at step 500 of 1500 planned (loss 2.00 → 1.18,
token accuracy 62.1% → 74.4% — see `trainer_state.json`).

- **Training regime:** bf16 mixed precision, 4-bit quantized base

#### Speeds, Sizes, Times

~2 hours on an RTX 4060 Laptop (8 GB VRAM, ~5 GB used). Adapter size 57 MB.

## Evaluation

14-prompt quality check across 5 categories (TTP, rules, tool reference,
KB, classification): 100% pass — samples in `SecGPT-Prod/doc.md`. 7-prompt
3-model comparison in `SecGPT-Prod/BENCHMARK.md` (7/7 vs 0/7 for the project's
from-scratch and GPT-2 models).

## Environmental Impact

- **Hardware Type:** NVIDIA RTX 4060 Laptop (8 GB)
- **Hours used:** ~2 (training)

## Technical Specifications

### Compute Infrastructure

#### Software

- PEFT 0.20.0, TRL 1.9.2, Transformers 5.8.1, PyTorch 2.11.0+cu128, bitsandbytes (NF4)

### Framework versions

- PEFT 0.20.0
