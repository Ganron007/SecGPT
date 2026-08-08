<div align="center">
  <img src="assets/logo.svg" alt="SecGPT logo" width="720"/>

  # SecGPT

  **Building a Cybersecurity LLM from Scratch to Production**

  [![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c)]()
  [![Base Model](https://img.shields.io/badge/base-Qwen2.5--3B-7b61ff)]()
  [![Benchmark](https://img.shields.io/badge/benchmark-291_prompts-blue)](SecGPT-Prod/eval/)
  [![License](https://img.shields.io/badge/license-educational%2Fresearch-lightgrey)](LICENSE)

  A comparative study of four approaches to building a domain-specific security
  language model on consumer hardware (RTX 4060 Laptop, 8 GB VRAM).
</div>

> [!NOTE]
> **Feature testing in progress.** SecGPT-Prod (Qwen2.5-3B + QLoRA) is far enough along for research demos and the 291-prompt benchmark, but training recipes, eval harness, and “from scratch → production” paths are still being validated. Expect data layouts, checkpoints, and docs to keep evolving.

---

## Table of Contents

- [The Question](#the-question)
- [The Four Models](#the-four-models)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Quick Start (SecGPT-Prod)](#quick-start-secgpt-prod)
- [Data Availability & Reproducing](#data-availability--reproducing)
- [Benchmark](#benchmark)
- [Hardware Requirements](#hardware-requirements)
- [Datasets Used](#datasets-used)
- [What's Next](#whats-next)
- [License](#license)
- [Author](#author)

## The Question

> Can you build a useful cybersecurity assistant locally, and what does the journey from "random weights" to "accurate answers" actually look like?

## The Four Models

| | SecGPTv2 | SecGPTv2.5 | SecGPTv3 | SecGPT-Prod |
|---|---|---|---|---|
| **Approach** | From scratch | From scratch (max effort) | Pretrained GPT-2 | Pretrained Qwen2.5-3B + QLoRA |
| **Params** | 17.4M | 98M | 124M | 1.7B active (3B total) |
| **Pretrain corpus** | 77.8 MB (25M tok) | 333 MB (108.7M tok, 16K BPE) | — (OpenAI) | — (Alibaba) |
| **SFT data** | 600 pairs (v1 corpus) | 23,746 pairs (v3) | 23,746 pairs (v3) | 23,746 pairs (v3) |
| **Pipeline** | Pretrain → SFT → DPO | Pretrain ✅ → SFT → DPO | SFT → DPO | SFT → DPO |
| **291-prompt benchmark** | legacy only | **18.2%** SFT / 7.6% DPO | **37.8%** SFT / 33.7% DPO | **58.8%** SFT+DPO (best) |
| **Headline** | Pipeline mechanics | The honest from-scratch ceiling | Format learned, knowledge can't be held | Working security assistant |

All models train on the **same v3 dataset** (23,746 pairs: STIX-verified MITRE,
real StackExchange Q&A, open KB, dedup'd rules, hard negatives) so the final
comparison isolates origin (scratch vs pretrained) and scale.

## Key Findings

1. **From-scratch training teaches the pipeline, not the product.** SecGPTv2 demonstrates every concept (tokenization, attention, loss, generation) but can't produce useful output at 17M params.

2. **Pretrained base + tiny data = failure.** GPT-2 Small with 600 SFT pairs collapsed into repetition. You need BOTH a capable base AND sufficient domain data.

3. **QLoRA on a strong base + 31K pairs = working product.** Qwen2.5-3B with 4-bit quantization + LoRA adapters writes valid Sigma rules and answers security questions in 5 GB VRAM — but the 291-prompt benchmark exposed the limits the 7-prompt demo hid.

4. **The 3-stage pipeline (Pretrain → SFT → Align) is universal.** Same pattern whether you're training from scratch or fine-tuning a 175B model. Scale changes, concepts don't.

5. **Naive SFT teaches behavior but corrupts facts (control experiment).** Base Qwen hallucinates MITRE IDs 20% of the time; after template-generated SFT (truncated 600-char answers) it hallucinates 87.5% — while genuinely improving at rules (+48 pts), classification (+38), and extraction (+30). Data quality is the bottleneck, not scale or alignment. See [SecGPT-Prod/eval/README.md](SecGPT-Prod/eval/README.md).

6. **DPO is accuracy-neutral.** Style alignment learned cleanly (99.3% reward accuracy) but moved no accuracy metric — hallucination is a knowledge problem, not a preference problem.

7. **Data quality beats data quantity (v1 → v3 iterations).** Template/truncated v1 data: 87.5% TTP hallucination. Mechanics-fixed v2.1: 76.2%. Knowledge-anchored v3 (STIX-verified IDs, real Q&A, hard negatives): **65.2%**. Each corpus generation bought factual grounding.

8. **Fair GPT-2 comparison (same v3 data):** 37.8% overall vs Qwen's 58.8% — but GPT-2 scores **94% on rules**, beating Qwen's 74%. Small pretrained models master format; they can't hold knowledge. (Bonus finding: the original GPT-2 "collapse" was partly a double label-shift bug in our own training code — documented in the repo.)

9. **Implementation is infrastructure.** Naive materialized-mask attention ran the 98M model at 1.3K tok/s; swapping to SDPA gave 24.6K tok/s (19×) — the difference between a 42-hour and a 2-hour pretrain.

## Repository Structure

```
SecGPT/
├── README.md              ← this file
├── LICENSE                ← educational/research use terms
├── DATA.md                ← data availability + reproduction pipelines
├── Docs/
│   ├── PLAN.md            ← master plan: 4-model study, status, sequence
│   └── V3_DATA_SPEC.md    ← next data iteration spec (STIX, StackExchange, open corpus)
├── assets/logo.svg        ← project logo
├── test_prompts.json      ← 80 legacy demo prompts (10 per category)
│                            (v2-style <|tag|> prefixes; superseded by SecGPT-Prod/eval/)
│
├── SecGPTv2/              ← From-scratch 17.4M (learning exercise, frozen)
│   ├── llm_build.md       ← Full build log covering all 8 pre-training steps
│   ├── requirements.txt
│   ├── src/               ← Training scripts (custom GPT, tokenizer, SFT, DPO)
│   ├── stage1_pre-training/  ← 8-step pipeline (input/output dirs per step)
│   ├── stage2_sft/        ← SFT documentation + results
│   └── stage3_alignment/  ← DPO documentation + results
│
├── SecGPTv2.5/            ← From-scratch 98M (max effort on 8 GB VRAM)
│   ├── src/               ← Corpus/tokenizer/model/train/SFT/DPO/eval scripts
│   ├── stage1_pre-training/  ← 333 MB corpus, 16K BPE, 24K-step pretrain ✅
│   ├── stage2_sft/        ← SFT on v3 data (next)
│   └── stage3_alignment/  ← DPO (pending)
│
├── SecGPTv3/              ← GPT-2 124M (fairness run on v3 data)
│   ├── doc.md             ← Original attempt analysis + lessons
│   ├── requirements.txt
│   └── src/               ← sft_v3.py / dpo_v3.py (37.8% / 33.7% benchmarked)
│
└── SecGPT-Prod/           ← Working model (Qwen2.5-3B + QLoRA)
    ├── doc.md             ← Build documentation (metrics, architecture, locations)
    ├── BENCHMARK.md       ← Legacy 7-prompt 3-model comparison
    ├── USAGE.md           ← How to run and use the model
    ├── eval/              ← 291-prompt benchmark harness + stage history
    ├── requirements.txt
    ├── src/               ← Data builders, QLoRA SFT, DPO, benchmark runner
    ├── stage1_sft/        ← SFT LoRA checkpoints (weights gitignored)
    └── stage2_alignment/  ← DPO checkpoint (weights gitignored)
```

## Quick Start (SecGPT-Prod)

```bash
pip install -r SecGPT-Prod/requirements.txt
cd SecGPT-Prod
python src/quality_check.py   # needs the trained LoRA adapter — see DATA.md
```

See [SecGPT-Prod/USAGE.md](SecGPT-Prod/USAGE.md) for full instructions.

## Data Availability & Reproducing

Model weights and generated datasets are not tracked in git (too large / regenerable / non-redistributable) — a fresh clone contains code, configs, and docs only.

See [DATA.md](DATA.md) for the annotated folder map and full reproduction pipelines.

## Benchmark

The old 7-prompt demo ([BENCHMARK.md](SecGPT-Prod/BENCHMARK.md)) is superseded by a
291-prompt two-layer harness ([SecGPT-Prod/eval/](SecGPT-Prod/eval/)) with objective
scorers, leakage splits, hallucination tracking, and N-way stage comparison.

Current stage history (291-prompt harness, full tables in `SecGPT-Prod/eval/`):

| Model / stage | Overall | TTP hallucination | Note |
|---|---|---|---|
| Qwen base (control) | 44.7% | 20.0% | pretrained honesty |
| Qwen SFT v1 data (31K, truncated) | 62.2% | 87.5% | behavior learned, facts corrupted |
| Qwen SFT v2.1 data | 58.4% | 76.2% | mechanics fixed |
| Qwen SFT v3 data | 57.0% | 65.2% | knowledge-anchored |
| **Qwen SFT v3 + DPO** | **58.8%** | 66.7% | **current Prod reference** |
| GPT-2 SFT v3 data | 37.8% | 0.0% | rule 94% (best of all models) |
| GPT-2 SFT v3 + DPO | 33.7% | 62.5% | DPO hurts accuracy again |
| SecGPTv2.5 (98M scratch) | 18.2% | 0.0% | classification 82% (matches Qwen); DPO destructive (7.6%) |

**Prompt sample:** *"Write a Sigma detection rule for suspicious PowerShell encoded command execution."*

| SecGPTv2 (17.4M) | SecGPTv3 (124M) | SecGPT-Prod (1.7B) |
|---|---|---|
| Format correct, content garbled | Repetitive garbage | ✅ Valid detection logic |

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

- [x] Security-specific evaluation framework → `SecGPT-Prod/eval/` (291 prompts, N-way stage history)
- [x] DPO alignment on SecGPT-Prod → accuracy-neutral, hallucination −4.2 pts
- [x] Data-quality iterations v2/v2.1/v3 → hallucination 87.5% → 65.2% ([spec](Docs/V3_DATA_SPEC.md))
- [x] SecGPTv3 fairness run: GPT-2 + SFT/DPO on v3 data (label-shift bug found & fixed)
- [x] SecGPTv2.5 pretrain: 98M from scratch, 108.7M tokens, val loss 1.73
- [x] SecGPTv2.5 SFT + DPO on v3 data + benchmark → 18.2% / 7.6% (DPO destructive at 98M)
- [x] Final 4-model verdict → [Docs/FINAL_VERDICT.md](Docs/FINAL_VERDICT.md)
- [ ] Scale to Qwen2.5-7B for better reasoning
- [ ] Multi-turn conversation support
- [ ] RAG integration (post-cutoff facts only; corpus already baked into weights)

## License

Educational/research use — see [LICENSE](LICENSE). Base models subject to their respective licenses:
- Qwen2.5-3B-Instruct: [Qwen License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE)
- GPT-2: OpenAI Modified MIT License

## Author

Built as part of the HTB AI Red Teamer certification study — learning by building.
