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
| SFT-500 (v2.1 data) | `qwen_qlora_v2_1/checkpoint-500` | 58.4% | 75.5% | 48.6% | **76.2%** | `sftv2_1_20260805_0300.json` |

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

```
                        qwenbase      sft500         dpo       sftv2     sftv2_1
classification             44.0%       82.0%       82.0%       84.0%       80.0%
consistency                50.0%       50.0%       50.0%       50.0%       50.0%
forensic_interp            40.0%       50.0%       40.0%       40.0%       70.0%
kb                         85.0%       85.0%       85.0%       80.0%       80.0%
ref                        44.0%       34.0%       28.0%       38.0%       34.0%
rule                       24.0%       72.0%       70.0%       28.0%       72.0%
rule_from_scenario         20.0%      100.0%       90.0%       50.0%       40.0%
soc_triage                100.0%      100.0%      100.0%       93.3%       86.7%
ttp                        26.0%       28.0%       28.0%       30.0%       26.0%
ttp_extract                30.0%       60.0%       60.0%       40.0%       50.0%

OVERALL                    44.7%       62.2%       60.1%       52.2%       58.4%
HELD-OUT                   58.5%       79.2%       79.2%       76.4%       75.5%
RECALL                     36.8%       52.4%       49.2%       38.4%       48.6%
TTP halluc.                20.0%       87.5%       83.3%       95.8%       76.2%
```

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
