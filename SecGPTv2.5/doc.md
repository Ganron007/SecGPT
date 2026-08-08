# SecGPTv2.5 — Build Documentation (98M from scratch)

> The maximum honest from-scratch effort on an RTX 4060 (8 GB VRAM), and the
> study's answer to: *what does it actually take to build a small/mid LLM,
> and why is it still limited?*

## Architecture

| Parameter | Value | Why |
|---|---|---|
| Layers × heads × width | 12 × 12 × 768 | GPT-2-small class; fits 8 GB with room for SFT/DPO |
| Params | 98.1M (weight-tied embeddings) | v2's 17.4M scaled 5.7× |
| Context | 512 (v2: 256) | longer answers in v3 data |
| Vocab | 16,384 BPE (v2: 8,000) | 8K under-compressed: 2.92 chars/token achieved |
| Attention | `F.scaled_dot_product_attention` (is_causal) | see "the 19× lesson" below |
| Precision | fp16 AMP + GradScaler | 8 GB budget |

**Alternatives considered:** 300M params (aggressive ceiling) — rejected:
~3× train time for marginal quality at our token budget, and it would crowd
the 8 GB needed for SFT/DPO/benchmark on the same card. RoPE/SwiGLU —
rejected for comparability with v2's architecture family; the study isolates
scale and data, not architectural tricks.

## Stage 1 — Pretraining ✅

**What:** 24,000 steps × batch 16 × 512 = ~197M token-passes over a 108.7M-token corpus (~1.8 epochs). LR 4e-4 → 4e-5 cosine, 500 warmup, grad-clip 1.0.

**Corpus (the v2 fix):** v2 pre-trained on 77.8 MB built from an extraction
where only 22% of CADRE chunks were even eligible. v2.5 rebuilds from the
section-based G: re-extraction (`kb_v3.jsonl`, 451K clean chunks):

| Source | Docs | Tag |
|---|---|---|
| kb_v3 (stratified, 30K/collection cap) | 171,461 | kb |
| NSL-KDD | 40,000 | net |
| DFIR-Nexus ttp/rule/ref sources | 15,826 | ttp/rule/ref |
| UCI SMS | 5,518 | spam/ham |
| **Total** | **232K docs, 333 MB, 108,688,552 tokens** | |

**Results:** final val loss **1.73** (random guess ln(16384) = 9.70).
Train time: ~6h total across runs at 10.7K tok/s sustained.

**The 19× lesson (infrastructure is a finding):** the first run used v2's
naive attention (materialized 512×512 causal mask per head per layer):
**1.3K tok/s, 7.9 GB VRAM → 42h projected**. Swapping to SDPA: **24.6K tok/s,
5.96 GB → same run in ~2h**. GPU was never the bottleneck; the kernel choice
was. (`src/speed_test.py` reproduces the measurement.)

## Stage 2 — SFT ✅

**What:** 1,500 steps on the shared v3 dataset (23,746 pairs, tag-formatted
`<|tag|>\nQ: …\nA: …`), LR 5e-5, batch 8 × 512, best-val checkpoint tracking.

**Why 5e-5:** v2 overfit at 1e-4 (train 0.16 / val 2.18). Bigger model,
lower LR, plus val-tracked saving instead of last-step saving.

**Results:** 6.6 min. Best val **2.05** (higher than pretrain val 1.73 is
expected — Q&A format is a new task). Benchmark: **17.9% overall**.

## Stage 3 — DPO ✅

**What:** 500 steps, β=0.3, batch 4 pairs/step, LR 1e-5, on 2,829 degraded
pairs (truncate/shuffle/de-structure/hedge — same strategies as every line).
β=0.3 and batched updates are the direct fixes for the original GPT-2 DPO
collapse (β=0.1, batch=1).

**Results:** 2 min, reward accuracy 100%, loss 0.023. Benchmark: **7.6%
overall — DPO was destructive** (recall split 4.3% → 0%). Consistent with
GPT-2: preference optimization needs a model that already holds the
distribution. At 98M it teaches the model to abandon what little it learned.

## Benchmark summary

| Stage | Overall | Held-out | Recall | TTP halluc. |
|---|---|---|---|---|
| SFT | 17.9% | 41.5% | 4.3% | 0.0% |
| SFT + DPO | 7.6% | 20.8% | 0.0% | 0.0% |

Bright spot: **classification 82% at SFT — matches Qwen-3B**. Simple
input→label tasks are learnable from scratch at this scale; factual
grounding is not. Full analysis: [Docs/FINAL_VERDICT.md](../Docs/FINAL_VERDICT.md).

## Reproduce

```powershell
cd SecGPTv2.5
python src/build_corpus.py          # 333 MB corpus (needs SecGPT-Prod/data/v3/kb_v3.jsonl)
python src/tokenizer.py             # 16K BPE, encodes + splits
python src/model.py                 # validate: 98.1M params
python src/train.py --steps 24000   # pretrain (~2h with SDPA; --resume supported)
python src/build_sft_corpus.py      # v3 pairs -> tag format
python src/sft_train.py --steps 1500
python src/build_dpo_corpus.py
python src/dpo_train.py --steps 500
python src/eval_v25.py --checkpoint stage2_sft/output/checkpoint_sft.pt --name v25sft
```

Status: ✅ complete (2026-08-08).
