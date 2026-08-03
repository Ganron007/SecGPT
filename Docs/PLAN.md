# SecGPT — Master Plan

> **Purpose:** a learning study, not a competition. Four models, one benchmark,
> every decision documented. We do our best and share the results — including
> (especially) the failures.

## The Four Models

| Model | Origin | Params | Meaning in the study |
|---|---|---|---|
| **SecGPTv2** | from scratch | 17.4M | Pipeline mechanics exist, capacity doesn't |
| **SecGPTv2.5** | from scratch | ~100M | Maximum honest from-scratch effort on 8 GB VRAM |
| **SecGPTv3** | GPT-2 pretrained | 124M | Borrowed fluency, fair-data retry |
| **SecGPT-Prod** | Qwen2.5 pretrained | 3B | The working product — reference point |

The central question v2.5 answers: *what does it actually take to build a
small/mid LLM from scratch, and why is it still limited without the right
training data at the right scale?* Bad benchmark numbers are expected and are
themselves the result — they quantify why pretraining scale matters.

## Pipeline (every model, every applicable stage)

```
Pretrain → SFT → DPO (alignment)
```

- **Pretrain:** raw corpus → base language ability (v2/v2.5 only; GPT-2/Qwen ship pretrained)
- **SFT:** 31,111 security Q&A pairs (`SecGPT-Prod/data/sft_32k.jsonl`, seed 42)
- **DPO:** preference pairs (structured > verbose), TRL DPOTrainer, β=0.3

## Current status

| Model | Pretrain | SFT | DPO | Benchmarked | Remaining |
|---|---|---|---|---|---|
| SecGPTv2 | ✅ 20K steps | ✅ 1500 (overfit 0.16/2.18) | ✅ 500 (0.69→0.07) | ⬜ | benchmark run + results chapter |
| SecGPTv2.5 | ⬜ | ⬜ | ⬜ | ⬜ | **everything** |
| SecGPTv3 | ✅ OpenAI | 🟨 600 pairs (unfair) | 🟨 collapsed | ⬜ | SFT-31K → DPO fairness run |
| SecGPT-Prod | ✅ Alibaba | ✅ 500/1500 steps | ⬜ | ⬜ | DPO + benchmark baseline |

## SecGPTv2.5 design (from scratch, max effort on RTX 4060 8 GB)

**Sizing math** (~16 B/param for bf16 weights+grads+fp32 Adam states, plus activations):

| Params | Memory | Verdict |
|---|---|---|
| 30M | ~0.5 GB | trivial, underwhelming |
| **~100M** | ~1.6 GB + ~2 GB activations | **chosen — GPT-2-small class** |
| ~300M | ~4.8 GB + ~2-3 GB | possible, OOM risk, marginal gain |

**Why 100M:** exact size parity with GPT-2 Small makes the comparison write
itself — our scratch model vs OpenAI's pretrain at the same parameter count.

- **Corpus:** 77.8 MB → **400-500 MB** (~150-200M tokens). Source already on disk:
  `cadre_kb.jsonl` (1.75 GB, v2 used only 30K of 483K chunks) + DFIR-Nexus + KDD + malimg.
  Data, not params, is the binding constraint (Chinchilla-optimal for 100M ≈ 2B tokens —
  unreachable; we document the gap and its effect).
- **Tokenizer:** retrain BPE at 16K vocab (v2's 8K under-compresses).
- **Architecture:** v2's custom GPT scaled: ~12 layers × 12 heads × 768d, context 512,
  weight tying, pre-norm. (Alternatives — RoPE/SwiGLU — documented at build time.)
- **Training:** fp16 AMP, cosine LR, ~8-16 h on the 4060 (overnight runs).
- **Then:** SFT (31K pairs) → DPO, same as every model.

## Fairness rule

Every model gets its best shot at every applicable stage before the final
comparison. GPT-2's original run used 600 SFT pairs vs Qwen's 31,111 — the one
genuinely unfair spot. Fix: **SecGPTv3 SFT on the same 31K pairs, then DPO**
(~1.5 h GPU). v2 already completed all 3 stages; Prod needs DPO.

## Benchmark (two layers, one harness)

Location: `SecGPT-Prod/eval/` (sets gitignored, results tracked).

**Layer 1 — accuracy** (`eval_set.jsonl`, ~250 prompts, built by `src/build_eval_set.py`):
- Held-out, leakage-checked: any source record present in `sft_32k.jsonl` is excluded
- Categories: ttp 50 / rule 50 / ref 50 / kb 40 / classification 50
- Objective scorers, no LLM judge: Sigma YAML validity, MITRE ID existence +
  description-ID match (**hallucination rate**), entity/keyword coverage,
  classification accuracy

**Layer 2 — practical utility** (`practical_set.jsonl`, built by `src/build_practical_set.py`):
scenario tasks, not Q&A — SOC alert triage, Sigma rule from prose scenario,
TTP extraction from threat-report paragraphs, forensic artifact interpretation,
consistency (same question × 3 phrasings → agreement), deployment metrics
(tok/s, VRAM, load time) measured by the runner.

**Runner** (`src/eval.py`): model-agnostic (any HF checkpoint ± LoRA; v2 needs a
custom adapter), batched greedy generation, per-category + overall scores,
JSON results, `--compare A B` stage-to-stage diff tables.

## Execution sequence

1. **Benchmark harness** ✅ done — `SecGPT-Prod/eval/` (240 accuracy + 51 practical,
   objective scorers, leakage splits, `--compare`). Baseline `sft500`: 62.2% overall,
   79.2% held-out, **87.5% TTP hallucination** — findings in `SecGPT-Prod/eval/README.md`
2. v2.5 corpus + tokenizer (CPU)
3. v2.5 pretrain (~8-16 h GPU, overnight)
4. SecGPTv3 SFT-31K + DPO (~1.5 h GPU)
5. SecGPT-Prod DPO (~3 h GPU)
6. v2.5 SFT + DPO
7. All 4 models through the benchmark → final 4-model `BENCHMARK.md`

## Documentation standard

Every model documents every stage: **what** (config, numbers) / **why**
(reasoning) / **how** (commands, code path) / **alternatives considered** /
**tradeoffs** / **results + failure analysis**. A reader must be able to
rebuild any model from its docs alone. v2.5 is documented live during the
build — nothing reconstructed from memory.

| Model | Docs |
|---|---|
| SecGPTv2 | `SecGPTv2/llm_build.md` + `stage2_sft/doc.md` + `stage3_alignment/doc.md` |
| SecGPTv2.5 | `SecGPTv2.5/` docs written during build |
| SecGPTv3 | `SecGPTv3/doc.md` (+ 31K fairness chapter) |
| SecGPT-Prod | `SecGPT-Prod/doc.md` + `USAGE.md` |
| Benchmark | `SecGPT-Prod/eval/` + results history |
