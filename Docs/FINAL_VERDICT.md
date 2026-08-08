# Final Verdict — 4 Models × 2 Stages

**Date:** 2026-08-08 · **Benchmark:** 291-prompt two-layer harness
(`SecGPT-Prod/eval/`) · **Hardware:** RTX 4060 Laptop, 8 GB VRAM

Every model line completed the same two post-pretrain stages — **SFT then
DPO — on the identical v3 dataset** (23,746 pairs: STIX-verified MITRE,
3,737 real StackExchange Q&A, open KB, dedup'd rules, hard negatives).
That control is what makes the comparison fair: the only variables left are
**origin** (scratch vs pretrained) and **scale**.

## The table

| Model line | Params | Origin | SFT | + DPO | Δ DPO |
|---|---|---|---|---|---|
| SecGPTv2.5 | 98M | **scratch** (333 MB corpus, 108.7M tokens) | 18.2% | 7.6% | **−10.6 (destructive)** |
| SecGPTv3 (GPT-2) | 124M | OpenAI pretrained | 37.8% | 33.7% | −4.1 |
| SecGPT-Prod (Qwen2.5-3B) | 3B | Alibaba pretrained | 57.0% | **58.8%** | **+1.8 (only winner)** |

Reference points: raw Qwen base 44.7% (20% TTP hallucination); Qwen v1-data
SFT 62.2% (87.5% hallucination — see the data-quality chapter below).

## Per-category highlights

| Category | v2.5 SFT | GPT-2 SFT | Qwen SFT+DPO |
|---|---|---|---|
| classification | **82%** (matches Qwen!) | 44% | 80% |
| rule | 0% | **94% (global best)** | 74% |
| kb | 7.5% | 17.5% | 55% |
| ref | 4% | 14% | **54%** |
| ttp | 4% | 10% | 28% |
| TTP hallucination | 0% (rarely asserts IDs) | 0% → 62.5% after DPO | 66.7% |

## Findings

1. **Pretraining is the moat.** A 98M model built from scratch on 108.7M
   tokens — a genuine max-effort run on this hardware — scores 18.2%.
   GPT-2 (124M, same size class) starts from OpenAI's WebText and scores
   37.8% with identical fine-tuning. Qwen-3B reaches 58.8%. The ordering is
   exactly origin × scale. Nothing about the fine-tuning recipe rescues a
   weak foundation.

2. **Scratch models still learn *tasks*.** v2.5's 82% classification matches
   Qwen — simple input→label mappings are learnable from 108M tokens. What
   isn't learnable: factual grounding (kb 7.5%, ttp 4%), which requires
   either far more tokens or a pretrained base.

3. **Small models master format, not knowledge.** GPT-2's 94% on rules beats
   every Qwen checkpoint — Sigma structure is pattern-matching on training
   formats. But its kb/ttp scores show it can't hold the content behind the
   format.

4. **DPO needs capacity.** DPO helped only the 3B model (+1.8). It was
   neutral-to-harmful at 124M (−4.1) and destructive at 98M (−10.3, recall
   split 4.3% → 0%). Preference optimization on a model that hasn't learned
   the underlying distribution just teaches it to abandon what little it had.
   The original GPT-2 "collapse" mystery is now fully explained: part
   label-shift bug (ours), part genuine small-model DPO fragility.

5. **Data quality was the lever inside a line; pretraining is the lever
   between lines.** Within Qwen, three corpus generations moved hallucination
   87.5% → 76.2% → 65.2% (see eval/README.md). Between lines, no data
   recipe closed more than a few points.

## Study artifacts

- Stage history + all result JSONs: `SecGPT-Prod/eval/results/`
- v2.5 build doc: `SecGPTv2.5/doc.md`
- GPT-2 fairness run + label-shift bug: `SecGPTv3/doc.md`
- Qwen line + data iterations: `SecGPT-Prod/doc.md`, `eval/README.md`
- Data spec: `Docs/V3_DATA_SPEC.md` · Master plan: `Docs/PLAN.md`
