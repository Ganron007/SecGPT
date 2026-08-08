# SecGPT-Prod Benchmark

Two-layer benchmark harness. Replaces the old 7-prompt demo with objective,
leakage-aware, stage-comparable measurement.

## Layers

| Layer | File | Size | What it measures |
|---|---|---|---|
| Accuracy | `eval_set.jsonl` | 240 prompts | Knowledge recall + generalization per category |
| Practical | `practical_set.jsonl` | 51 scenarios | Real-world usefulness (triage, extraction, consistency) |

**Leakage policy:** DFIR-Nexus sources were fully consumed by training, so
their prompts measure trained-knowledge *recall*; kb/SMS/KDD have held-out
records and measure *generalization*. Every item is flagged `meta.leaked`;
results report both splits.

## Scorers (rule-based, no LLM judge)

| Category | Scorer |
|---|---|
| ttp | MITRE ID correctness + keyword coverage; tracks **hallucination rate** (invented/misattributed IDs) |
| rule | Canonical Sigma YAML schema **or** flat training format (`condition:`/`selection:`); keyword fallback |
| ref / forensic_interp | Entity mention + source keyword coverage |
| kb | Source keyword coverage |
| classification | Exact label match (spam/ham, normal/attack type) |
| soc_triage | Rubric: severity + verdict + recommended action (2 of 3 groups) |
| ttp_extract | ID recall vs source text, ≤1 extra invented |
| consistency | 3 phrasings → pairwise containment agreement ≥ 0.4 |

## Usage

```powershell
cd SecGPT-Prod

# rebuild the sets (requires ../SecGPTv2/data sources)
python src/build_eval_set.py
python src/build_practical_set.py

# run a checkpoint (batched greedy decoding, ~35 min on RTX 4060)
python src/eval.py --name sft500
python src/eval.py --lora stage2_alignment/output/.../checkpoint-XXX --name dpo
python src/eval.py --model openai-community/gpt2 --no-lora --name gpt2-sft

# smoke test (2 prompts per category, ~2 min)
python src/eval.py --limit 2 --name smoke

# stage-to-stage comparison
python src/eval.py --compare results/sft500_xxx.json results/dpo_xxx.json
```

Results land in `eval/results/<name>_<timestamp>.json` (tracked in git as the
project's progress history). Eval sets are gitignored (`*.jsonl`).

## Stage history — SecGPT-Prod (Qwen2.5-3B)

Standard ritual: **every model is benchmarked after every training stage**
(base control, SFT, DPO, scale, multi-turn). Results in `eval/results/`;
N-way comparisons via `python src/eval.py --compare a.json b.json c.json ...`.

| Stage | Checkpoint | Overall | Held-out | Recall | TTP halluc. | Result file |
|---|---|---|---|---|---|---|
| **Base (control)** | none (raw Qwen2.5-3B-Instruct) | 44.7% | 58.5% | 36.8% | **20.0%** | `qwenbase_20260804_0550.json` |
| SFT-500 (v1 data) | `qwen_qlora/checkpoint-500` | 62.2% | 79.2% | 52.4% | 87.5% | `sft500_20260803_2311.json` |
| SFT-500 + DPO | `qwen_dpo/final` | 60.1% | 79.2% | 49.2% | 83.3% | `dpo_20260804_0248.json` |
| SFT-500 (v2 data) | `qwen_qlora_v2/checkpoint-500` | 52.2% | 76.4% | 38.4% | 95.8% | `sftv2_20260804_2108.json` |
| SFT-500 (v2.1 data) | `qwen_qlora_v2_1/checkpoint-500` | 58.4% | 75.5% | 48.6% | 76.2% | `sftv2_1_20260805_0300.json` |
| SFT-500 (v3 data) | `qwen_qlora_v3/checkpoint-500` | 57.0% | 63.2% | 53.5% | **65.2%** | `sftv3_20260805_1807.json` |
| SFT-v3 + DPO | `qwen_dpo_v3/final` | 58.8% | 62.3% | 56.8% | 66.7% | `dpov3_20260805_2325.json` |
| GPT-2 SFT-v3 | `SecGPTv3/stage2_sft/output/model_v3` | 37.8% | 32.1% | 41.1% | 0.0% | `gpt2sftv3b_20260806_2051.json` |
| GPT-2 SFT-v3 + DPO | `SecGPTv3/stage3_alignment/output/model_dpo_v3/final` | 33.7% | 26.4% | 37.8% | 62.5% | `gpt2dpov3b_20260806_2104.json` |
| SecGPTv2.5 SFT (98M scratch) | `SecGPTv2.5/stage2_sft/output/checkpoint_sft.pt` | 18.2% | 42.5% | 4.3% | 0.0% | `v25sft_20260808_0920.json` |
| SecGPTv2.5 SFT + DPO | `SecGPTv2.5/stage3_alignment/output/checkpoint_dpo.pt` | 7.6% | 20.8% | 0.0% | 0.0% | `v25dpo_20260808_0921.json` |

### Final cross-model verdict (2026-08-08)

All four model lines completed SFT + DPO on the identical v3 dataset
(23,746 pairs). See [Docs/FINAL_VERDICT.md](../../Docs/FINAL_VERDICT.md).

| Line | Params | Origin | SFT | +DPO | Best category |
|---|---|---|---|---|---|
| SecGPTv2.5 | 98M | scratch (108.7M tokens) | 18.2% | 7.6% (DPO destructive) | classification 82% |
| SecGPTv3 (GPT-2) | 124M | OpenAI pretrained | 37.8% | 33.7% | **rule 94% (global best)** |
| SecGPT-Prod (Qwen) | 3B | Alibaba pretrained | 57.0% | **58.8%** | ref 54%, rule 74% |

### The v3 corpus experiment (2026-08-05)

v3 data (23,746 pairs): STIX-verified MITRE relationships, real StackExchange
Q&A (3,737 pairs), HackTricks/OWASP open KB, G:-re-extracted CADRE chunks,
v2.1's fixed rule/ref/classification.

**Delivered exactly what it was built for — factual grounding:**

| Metric | v1 | +DPO | v2.1 | v3 | +DPO-v3 |
|---|---|---|---|---|---|
| TTP hallucination | 87.5% | 83.3% | 76.2% | **65.2%** | 66.7% |
| ref | 34% | 28% | 34% | 44% | **54%** |
| rule | 72% | 70% | 72% | 70% | **74%** |
| rule_from_scenario | 100% | 90% | 40% | 90% | 90% |

**Trade-off:** kb 57.5% and consistency 16.7% dropped with v3 (DPO didn't
recover them). Likely cause: diverse answer styles (SE/open-KB) hurt the
keyword-overlap scorers — partly measurement artifact, partly real style
variance. forensic_interp dipped to 40% after DPO (n=10, one-item swings).

**Verdict:** hallucination fell 87.5% → 65.2% across the data generations
(87.5 → 76.2 → 65.2). Each corpus iteration bought grounding at some style
cost. `qwen_dpo_v3/final` is the current Prod reference: best ref (54%,
above base), best rule (74%), best recall split (56.8%), and the most honest
ID behavior of any trained checkpoint.

### The v2 → v2.1 experiments (2026-08-04/05)

**v2 scored worse (52.2%)** — root-caused to two builder defects: (1) all rule
sources unified under "Write a Sigma rule" templates, so the model answered
Sigma requests with Elastic EQL/Splunk SPL bodies; (2) the 1,072-record MITRE
pool was sampled ~4.3× each, teaching over-confident ID assertion
(hallucination 95.8%).

**v2.1 fixes worked partially:** per-source templates restored rule to 72%
(from 28%), disjoint MITRE slices + capped verification dropped hallucination
to **76.2% — best SFT run** (vs 87.5% v1). forensic_interp hit 70% (best ever).
But applied-task scores dipped: rule_from_scenario 40% (v1: 100%), soc_triage
86.7% (v1: 100%), ttp_extract 50% (v1: 60%).

**Interpretation:** v1's perfect practical scores benefited from near-identical
train/test formats (leakage-adjacent advantage); v2.1's longer, per-source
answers trade applied-format fluency for factual grounding. Data changes move
metrics in *different directions per category* — there is no free lunch, only
chosen trade-offs. Cumulative picture:

| Metric | v1 | v2 | v2.1 |
|---|---|---|---|
| Overall | 62.2% | 52.2% | 58.4% |
| TTP hallucination | 87.5% | 95.8% | **76.2%** |
| rule | 72% | 28% | 72% |
| forensic_interp | 50% | 40% | **70%** |

**Meta-lesson:** the benchmark caught in 35 minutes what eyeballing the data
did not. This is why the harness exists.

### Full comparison (current line)

| Category | base | sft500 (v1) | +DPO (v1) | sftv2.1 | sftv3 | +DPO (v3) |
|---|---|---|---|---|---|---|
| classification | 44.0% | 82.0% | 82.0% | 80.0% | 80.0% | 80.0% |
| consistency | 50.0% | 50.0% | 50.0% | 50.0% | 16.7% | 16.7% |
| forensic_interp | 40.0% | 50.0% | 40.0% | 70.0% | 70.0% | 40.0% |
| kb | 85.0% | 85.0% | 85.0% | 80.0% | 57.5% | 55.0% |
| ref | 44.0% | 34.0% | 28.0% | 34.0% | 44.0% | **54.0%** |
| rule | 24.0% | 72.0% | 70.0% | 72.0% | 70.0% | **74.0%** |
| rule_from_scenario | 20.0% | 100.0% | 90.0% | 40.0% | 90.0% | 90.0% |
| soc_triage | 100.0% | 100.0% | 100.0% | 86.7% | 86.7% | 93.3% |
| ttp | 26.0% | 28.0% | 28.0% | 26.0% | 26.0% | 28.0% |
| ttp_extract | 30.0% | 60.0% | 60.0% | 50.0% | 30.0% | 30.0% |
| **Overall** | 44.7% | 62.2% | 60.1% | 58.4% | 57.0% | **58.8%** |
| Held-out | 58.5% | 79.2% | 79.2% | 75.5% | 63.2% | 62.3% |
| Recall | 36.8% | 52.4% | 49.2% | 48.6% | 53.5% | 56.8% |
| TTP hallucination | 20.0% | 87.5% | 83.3% | 76.2% | **65.2%** | 66.7% |

### Findings

1. **SFT taught behavior** (real gains): rule +48, rule_from_scenario +80,
 classification +38, ttp_extract +30 over base.
2. **SFT corrupted factual grounding** (the corpus flaw): TTP hallucination
 20% → 87.5%; ref knowledge −10 vs base. Template pairs with truncated
 600-char answers taught confident, wrong ID↔description mappings.
3. **DPO was accuracy-neutral** (within noise) but slightly recovered honesty
 (87.5 → 83.3%). Style alignment can't fix a knowledge problem.
4. **Conclusion:** next iteration is data *quality* (complete, verified,
 anchored answers + hard negatives), not more scale or alignment.

## Baseline: SFT checkpoint-500 (Qwen2.5-3B + LoRA, 31K pairs)

Run: `sft500_20260803_2311.json` — 291 prompts, greedy, batch 8, 33.7 tok/s, peak 2.5 GB VRAM.

| Metric | Value |
|---|---|
| Overall pass | 62.2% |
| Held-out (generalization) | 79.2% |
| Recall (trained data) | 52.4% |
| classification | 82.0% |
| kb | 85.0% |
| rule | 72.0% (canonical YAML only 8% — model uses its flat training format) |
| ref | 34.0% |
| **ttp** | **28.0% — hallucination rate 87.5%** |
| soc_triage | 100% |
| rule_from_scenario | 100% |
| ttp_extract | 60% |
| forensic_interp | 50% |
| consistency | 50% |

### What the old 7/7 demo hid

1. **MITRE ID hallucination is rampant (87.5%).** At 7 prompts the model looked
 accurate; at 50 it invents or misattributes technique IDs most of the time.
2. **Specific factual recall (ttp/ref) is the weak spot** — generic categories
 (kb, triage) pass easily; precise IDs and tool details don't.
3. **Recall split < held-out split** — driven by category composition (the hard
 ttp/rule/ref categories are all in-training data), and by confident
 wrong-specific answers on trained topics vs safe generic answers elsewhere.

These findings define what DPO and the 7B scale-up must improve.
