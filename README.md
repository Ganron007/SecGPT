<div align="center">
  <img src="assets/logo.svg" alt="SecGPT logo" width="720"/>

  # SecGPT

  **Building a Cybersecurity LLM from Scratch to Production**

  [![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c)]()
  [![Base Model](https://img.shields.io/badge/base-Qwen2.5--3B-7b61ff)]()
  [![Benchmarks](https://img.shields.io/badge/benchmarks-7%2F7-brightgreen)]()
  [![License](https://img.shields.io/badge/license-educational%2Fresearch-lightgrey)](LICENSE)

  A comparative study of three approaches to building a domain-specific security
  language model on consumer hardware (RTX 4060 Laptop, 8 GB VRAM).
</div>

---

## Table of Contents

- [The Question](#the-question)
- [The Three Models](#the-three-models)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Quick Start (SecGPTv3)](#quick-start-secgptv3)
- [Data Availability & Reproducing](#data-availability--reproducing)
- [Benchmark Sample](#benchmark-sample)
- [Hardware Requirements](#hardware-requirements)
- [Datasets Used](#datasets-used)
- [What's Next](#whats-next)
- [License](#license)
- [Author](#author)

## The Question

> Can you build a useful cybersecurity assistant locally, and what does the journey from "random weights" to "accurate answers" actually look like?

## The Three Models

| | SecGPTv2 | SecGPT-Prod | SecGPTv3 |
|---|---|---|---|
| **Approach** | Built from scratch | Pretrained GPT-2 + SFT | Pretrained Qwen2.5-3B + QLoRA |
| **Params** | 17.4M | 124M | 1.7B |
| **Training data** | 77.8 MB corpus | 600 Q&A pairs | 31,111 Q&A pairs |
| **Pipeline** | Pretrain → SFT → DPO | Domain-adapt → SFT → DPO | QLoRA SFT |
| **Result** | Readable fragments | Broken (collapsed) | ✅ Accurate security answers |
| **Pass rate** | 0/7 benchmarks | 0/7 benchmarks | **7/7 benchmarks** |
| **Train time** | 51 min | 27 min | 2 hours |

## Key Findings

1. **From-scratch training teaches the pipeline, not the product.** SecGPTv2 demonstrates every concept (tokenization, attention, loss, generation) but can't produce useful output at 17M params.

2. **Pretrained base + tiny data = failure.** GPT-2 Small with 600 SFT pairs collapsed into repetition. You need BOTH a capable base AND sufficient domain data.

3. **QLoRA on a strong base + 31K pairs = working product.** Qwen2.5-3B with 4-bit quantization + LoRA adapters produces accurate MITRE descriptions, valid Sigma rules, and correct vulnerability analysis — all in 5 GB VRAM.

4. **The 3-stage pipeline (Pretrain → SFT → Align) is universal.** Same pattern whether you're training from scratch or fine-tuning a 175B model. Scale changes, concepts don't.

## Repository Structure

```
SecGPT/
├── README.md              ← this file
├── LICENSE                ← educational/research use terms
├── assets/logo.svg        ← project logo
├── test_prompts.json      ← 80 benchmark prompts (10 per category)
│                            (v2-style <|tag|> prefixes; strip tags for v3 chat prompts)
│
├── SecGPTv2/              ← From-scratch model (learning exercise)
│   ├── llm_build.md       ← Full build log covering all 8 pre-training steps
│   ├── requirements.txt
│   ├── src/               ← Training scripts (custom GPT, tokenizer, SFT, DPO)
│   ├── stage1_pre-training/  ← 8-step pipeline (input/output dirs per step)
│   ├── stage2_sft/        ← SFT documentation + results
│   └── stage3_alignment/  ← DPO documentation + results
│
├── SecGPT-Prod/           ← GPT-2 attempt (documented failure + lessons)
│   ├── doc.md             ← Honest analysis of what went wrong
│   ├── requirements.txt
│   └── src/               ← Pipeline scripts
│
└── SecGPTv3/              ← Final working model (Qwen2.5-3B + QLoRA)
    ├── doc.md             ← Build documentation (metrics, architecture, locations)
    ├── BENCHMARK.md       ← 7-prompt comparison across all 3 models
    ├── USAGE.md           ← How to run and use the model
    ├── requirements.txt
    ├── src/               ← Dataset generation, QLoRA training, quality check
    └── stage1_sft/        ← Trained LoRA checkpoint (weights gitignored, see below)
```

## Quick Start (SecGPTv3)

```bash
pip install -r SecGPTv3/requirements.txt
cd SecGPTv3
python src/quality_check.py   # needs the trained LoRA adapter — see below
```

See [SecGPTv3/USAGE.md](SecGPTv3/USAGE.md) for full instructions.

## Data Availability & Reproducing

Model weights and generated datasets are **gitignored** (too large / regenerable). A fresh clone contains only code, configs, and docs. Missing pieces and how to restore them:

| Missing (gitignored) | Needed by | How to restore |
|---|---|---|
| `SecGPTv3/stage1_sft/output/qwen_qlora/checkpoint-500/adapter_model.safetensors` (57 MB) — the trained LoRA | v3 inference | Retrain: steps below (~2 h on RTX 4060) |
| `SecGPTv3/data/sft_32k.jsonl` (31,111 pairs) | v3 training | `python src/build_sft_32k.py` (reads `SecGPTv2/data/`) |
| `SecGPTv2/data/cadre_kb.jsonl` (1.75 GB, 483K chunks) | v2 + v3 dataset builders | CADRE KB — proprietary corpus, not redistributable |
| `SecGPTv2/data/dfir_nexus_sources/*.jsonl` (23 files) | v2 + v3 dataset builders | DFIR-Nexus exports (MITRE, Sigma, LOLBAS, GTFOBins, etc.) |
| `SecGPTv2/data/KDD+.txt` (20 MB) | v2 corpus + v3 builder | NSL-KDD dataset (public) |
| `SecGPTv2/data/sms+spam+collection.zip` | v2 corpus + v3 builder | UCI SMS Spam Collection (public) |
| `SecGPTv2/data/malimg.zip` | v2 corpus | Malimg dataset (public; auto-extracted by `build_corpus.py`) |
| `SecGPTv2/stage1_pre-training/**` checkpoints, `SecGPT-Prod/**/model.safetensors` | v2/Prod inference | Retrain per `llm_build.md` / `SecGPT-Prod/doc.md` |

Without the proprietary CADRE KB and DFIR-Nexus exports the datasets cannot be regenerated at full size; the public sources (NSL-KDD, UCI SMS, Malimg) restore the classification portions.

### Full v3 reproduction (given the data above)

```bash
cd SecGPTv3
python src/build_sft_32k.py        # 1. dataset (~2 min)  → data/sft_32k.jsonl
python src/qlora_sft.py            # 2. QLoRA training (~2 h) → stage1_sft/output/qwen_qlora/
python src/quality_check.py        # 3. benchmark 14 prompts
```

## Benchmark Sample

**Prompt:** *"Write a Sigma detection rule for suspicious PowerShell encoded command execution."*

| SecGPTv2 (17.4M) | SecGPT-Prod (124M) | SecGPTv3 (1.7B) |
|---|---|---|
| Format correct, content garbled | Repetitive garbage | ✅ Valid Sigma rule with proper detection logic |

See [SecGPTv3/BENCHMARK.md](SecGPTv3/BENCHMARK.md) for full 7-prompt comparison.

## Hardware Requirements

- GPU: NVIDIA with 6+ GB VRAM (tested: RTX 4060 Laptop, 8 GB)
- RAM: 16+ GB
- Disk: ~10 GB (base model + dependencies)
- Python: 3.11+
- OS: Windows/Linux

## Datasets Used

| Source | Records | Content |
|---|---|---|
| DFIR-Nexus (MITRE, Sigma, LOLBAS, GTFOBins, etc.) | 17,950 | Detection rules, TTPs, tool references |
| CADRE KB (SANS, HTB Academy, Malpedia, Mandiant, etc.) | 483,800 chunks | Security courses, IR reports, malware descriptions |
| UCI SMS Spam Collection | 5,574 | Spam/ham classification |
| NSL-KDD | 148,517 | Network intrusion detection |

## What's Next

- [ ] DPO alignment on SecGPTv3 (prefer structured over verbose)
- [ ] RAG integration (retrieval over full corpus for factual grounding)
- [ ] Scale to Qwen2.5-7B for better reasoning
- [ ] Multi-turn conversation support
- [ ] Security-specific evaluation framework (beyond 7 prompts)

## License

Educational/research use — see [LICENSE](LICENSE). Base models subject to their respective licenses:
- Qwen2.5-3B-Instruct: [Qwen License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE)
- GPT-2: OpenAI Modified MIT License

## Author

Built as part of the HTB AI Red Teamer certification study — learning by building.
