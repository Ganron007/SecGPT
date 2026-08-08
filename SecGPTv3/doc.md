# Phase B — SecGPTv3: GPT-2 Small Pipeline (The Company Approach)

## What This Is

Taking a pretrained GPT-2 Small (124M params, already fluent in English) and running the same 3-stage pipeline as SecGPTv2. This is what companies do: start with a pretrained base, fine-tune for your domain.

**Model:** `openai-community/gpt2` (124,439,808 params, 50,257 BPE vocab, 1024 context)

---

## Results Summary

| Stage | Time | Loss | Outcome |
|---|---|---|---|
| Stage 1: Domain-adapt | 25 min | 9.4 → 2.6 | Failed: destroyed English fluency (custom tags confused tokenizer) |
| Stage 2: SFT (direct, no domain-adapt) | 68s | 9.7 → 4.0 | Learned format, produces fragments, falls into repetition |
| Stage 3: DPO | 65s | 0.69 → 0.16 | Failed: collapsed model (repetitive garbage) |

**Best checkpoint:** `stage2_sft/output/model_direct/` (direct SFT, no domain-adapt)

---

## What Worked

- Loss decreased properly with corrected eval + conservative LR (1e-5)
- Model recognizes Q→A format and attempts structured responses
- Fragments of correct content appear: "T1059 is a technique...", "Detect the execution of PowerShell...", "Status: test, Level: medium, Author:..."
- GPT-2's pretrained English knowledge partially survives fine-tuning

## What Didn't Work

| Problem | Cause | Fix needed |
|---|---|---|
| Domain-adapt destroyed fluency | `<\|tag\|>` tokens aren't in GPT-2's vocab — get split into garbage subwords | Strip tags, or add as special tokens |
| SFT produces repetition loops | 600 pairs too few for 124M model; structured content (YAML, Sigma) is hard | Need 5K-50K pairs |
| DPO collapsed model | Batch=1 per-sample updates too aggressive; beta too low | Use batch DPO, higher beta, fewer steps |
| Initial loss 9.7 (vs expected ~3.5) | Security rules/YAML are rare in GPT-2's WebText training | Expected — domain gap is real |

---

## Key Lesson: Why Companies Need Large SFT Datasets

| Dataset | Pairs | Model | Result |
|---|---|---|---|
| **Ours** | 600 | GPT-2 Small (124M) | Format learned, content garbled |
| Alpaca | 52,000 | LLaMA 7B | Fluent instruction-following |
| Vicuna | 70,000 | LLaMA 13B | Near-ChatGPT quality |
| Orca | 5,000,000 | LLaMA 2 13B | GPT-4-level reasoning |

**The ratio:** our 600 pairs for 124M params = 0.005 pairs per 1K params. Alpaca had 52K pairs for 7B params = 0.007 pairs per 1K params. Similar ratio, but Alpaca's pairs were *simple English tasks* while ours are *structured security rules* (much harder to learn).

**Bottom line:** to get useful output from GPT-2 Small on security Q&A, we'd need:
- 5,000–10,000 high-quality security Q&A pairs (not 600)
- OR a larger base model (LLaMA, Phi-3) that already knows more
- OR simpler tasks (summarization, classification) where 600 pairs suffices

---

## SecGPTv2 vs SecGPTv3: Honest Comparison

| | SecGPTv2 (from scratch) | SecGPTv3 (GPT-2 + SFT) |
|---|---|---|
| Params | 17.4M | 124M |
| Training data | 77.8 MB corpus | 600 Q&A pairs |
| English fluency | None (learned from scratch) | Pretrained (but degraded by SFT) |
| Domain knowledge | Strong (trained on 77 MB security text) | Weak (only saw 600 examples) |
| Q&A ability | Learned Q→A format (SFT stage) | Attempts answers, falls into loops |
| Structured output | Good (Sigma format, MITRE IDs) | Fragments of format, repetitive |
| Best output | Readable security fragments | Broken English with correct keywords |
| Total train time | ~51 min | ~27 min |

**Winner for domain content:** SecGPTv2 (trained on 130× more security data)
**Winner for English structure:** SecGPTv3 (pretrained base, when it doesn't loop)
**Neither is useful as a product** — both are learning exercises

---

## Input / Output Contract

| | Description | Location |
|---|---|---|
| **Input** | GPT-2 Small (HuggingFace) | `openai-community/gpt2` (downloaded to cache) |
| **Input** | SFT dataset (shared with SecGPTv2) | `../SecGPTv2/stage2_sft/output/sft_data.jsonl` |
| **Input** | DPO dataset (shared with SecGPTv2) | `../SecGPTv2/stage3_alignment/output/dpo_data.jsonl` |
| **Output** | Domain-adapted model (broken) | `stage1_domain-adapt/output/model/` |
| **Output** | SFT model (best checkpoint) | `stage2_sft/output/model_direct/` |
| **Output** | DPO model (collapsed) | `stage3_alignment/output/model/` |

---

## How to Reproduce

```bash
cd SecGPTv3

# Direct SFT (the working approach)
python src/sft_direct.py

# Test generation
python src/sft_direct.py --generate

# Full pipeline (domain-adapt + SFT + DPO — for reference, produces worse results)
python src/pipeline.py --stage 1
python src/pipeline.py --stage 2
python src/pipeline.py --stage 3
```

---

## What Would Actually Work (for a useful product)

1. **Larger base model:** Phi-3 Mini (3.8B) or LLaMA 3.2 (3B) — fits in 8 GB quantized
2. **Larger SFT dataset:** 5K-10K security Q&A pairs (generate with GPT-4/Claude from our corpus)
3. **Proper DPO:** batch processing, beta=0.3, 1000+ diverse preference pairs
4. **Or skip all this:** use RAG (retrieval-augmented generation) with a strong API model + our corpus as knowledge base

---

## Fairness Run (2026-08-06) — same v3 data as Qwen

The original run above used 600 pairs vs Qwen's 31K — an unfair comparison.
The fairness run re-trains GPT-2 on the **identical v3 dataset** (23,746
pairs) with the same recipe (500 steps, effective batch 16, block 512).

**Bug found:** the original scripts (and the first fairness attempt) passed
pre-shifted labels to HF GPT-2, which shifts labels *again* internally — a
double shift that trained the model on a corrupted 2-step objective (start
loss 9.5 where a healthy GPT-2 shows ~3.9; generation collapsed into
punctuation loops). The original "collapse" was therefore partly our bug.
Fixed in `src/sft_v3.py` (`labels=x`, model handles the shift).

| Stage | Overall | Held-out | Recall | TTP halluc. |
|---|---|---|---|---|
| SFT on v3 data | **37.8%** | 32.1% | 41.1% | 0.0% |
| SFT + DPO (β=0.3) | 33.7% | 26.4% | 37.8% | 62.5% |

**Findings:** with fair data GPT-2 produces real answers (no collapse).
**Rule 94% — the best rule score of any model in the study** (format is
pattern-matching, and GPT-2 memorizes it perfectly). But kb 17.5% / ttp 10%:
124M pretrained params can hold format, not this much knowledge — 21 points
below Qwen-3B on identical data. DPO again failed to help (−4.1), confirming
the small-model DPO fragility later seen at 98M. Details:
[Docs/FINAL_VERDICT.md](../Docs/FINAL_VERDICT.md).

Scripts: `src/sft_v3.py`, `src/dpo_v3.py` (TRL DPOTrainer, fp16, no PEFT).

## Status: Complete (2026-08-02) — documented as a learning exercise with honest limitations
