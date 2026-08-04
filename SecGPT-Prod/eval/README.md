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
(SFT, DPO, scale, multi-turn). Results in `eval/results/`, comparisons here.

| Stage | Checkpoint | Overall | Held-out | Recall | TTP halluc. | Result file |
|---|---|---|---|---|---|---|
| **Base (control)** | none (raw Qwen2.5-3B-Instruct) | 44.7% | 58.5% | 36.8% | **20.0%** | `qwenbase_20260804_0550.json` |
| SFT-500 | `qwen_qlora/checkpoint-500` | 62.2% | 79.2% | 52.4% | 87.5% | `sft500_20260803_2311.json` |
| SFT + DPO | `qwen_dpo/final` | 60.1% | 79.2% | 49.2% | 83.3% | `dpo_20260804_0248.json` |

### Base vs SFT — the control experiment (2026-08-04)

**SFT taught behavior but corrupted factual grounding.**

| Category | Base | SFT | Δ | Interpretation |
|---|---|---|---|---|
| classification | 44.0% | 82.0% | **+38** | SFT taught the task |
| rule | 24.0% | 72.0% | **+48** | SFT taught the format |
| rule_from_scenario | 20.0% | 100% | **+80** | biggest real gain |
| ttp_extract | 30.0% | 60.0% | **+30** | real gain |
| ref (LOLBAS/tools) | 44.0% | 34.0% | **−10** | SFT made it WORSE |
| **TTP hallucination** | **20.0%** | **87.5%** | **+67.5** | **SFT taught overconfident ID-citing** |
| kb / consistency / soc_triage / ttp | ≈equal | ≈equal | 0 | pretrained skills |

**The study's clearest finding:** template-generated SFT data (fixed question
templates + truncated 600-char source excerpts) teaches *task behavior*
(classify, write rules, extract) but *degrades knowledge* — the model learned
to answer in "ID + Description" format with confident, often wrong ID
attributions. Base Qwen is 4.4× more honest about technique IDs.

**Implication:** the next data iteration must be knowledge-anchored (complete,
correct ID↔description mappings, no mid-context truncation) rather than more
of the same. DPO slightly recovered honesty (87.5→83.3%).

### SFT vs DPO verdict (2026-08-04)

**Accuracy-neutral, slight hallucination improvement, pipeline healthy.**

- DPO training was clean: 99.3% reward accuracy, margin 3.12, no collapse
  (β=0.3, 1 epoch, 3,818 pairs, 147 min)
- Benchmark delta: −2.1 pts overall (within noise: largest single-category
  drops are 1–3 items), held-out identical at 79.2%
- TTP hallucination improved 87.5% → 83.3% (−4.2 pts) but remains the
  dominant weakness
- **Lesson:** DPO learned the *style* preference (structured > verbose) it was
  trained on, but the benchmark measures *accuracy* — and hallucination is a
  knowledge problem, not a preference problem. Fixing it needs better/more
  data (or 7B capacity), not more alignment.
- Decision: keep `qwen_dpo/final` as the Prod reference checkpoint (equal
  accuracy, marginally less hallucination, trained style preference).

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
