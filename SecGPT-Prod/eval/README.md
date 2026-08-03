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
