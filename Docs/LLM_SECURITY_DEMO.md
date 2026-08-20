# LLM Security Demonstration — Phase 2 Spec

> **Status:** in progress (Tier 1 scripts written, runs pending) · **Created:**
> 2026-08-08 · **Companion docs:** [PLAN.md](PLAN.md) (master plan),
> [FINAL_VERDICT.md](FINAL_VERDICT.md) (Phase 1 results),
> [SIBLING-DARKAI.md](SIBLING-DARKAI.md) (ownership rules vs DarkAI)

> [!IMPORTANT]
> **Scope of attacks.** Everything in this phase targets SecGPT's own
> locally-trained models (SecGPTv2.5 / SecGPTv3 / SecGPT-Prod). No external
> APIs, no production systems, no third-party models are touched. Attack
> artifacts stay model-internal: weights, log-probs, retraining, and
> generation on our checkpoints only.

**Standard basis:** OWASP GenAI LLM Top 10 2026, OWASP ML Top 10 2023
(draft) · **Threat mapping:** MITRE ATLAS · **Governance mapping:**
NIST AI RMF (Govern / Map / Measure / Manage)

## What this phase is (and is not)

| | SecGPT Phase 2 | DarkAI (sibling lab) |
|---|---|---|
| **Attacks the** | model **brain** (weights, training, inference internals) | model **body** (deployed AI applications) |
| **Targets** | our own 3 checkpoints | 12 containerized vulnerable AI apps |
| **Method** | PyTorch tensors, log-probs, retraining | HTTP endpoints, prompts, Docker |
| **Standard** | OWASP LLM/ML Top 10, ATLAS, NIST AI RMF | OWASP LLM 2025, ATLAS, SAIF, NIST AI 600-1 |

**This phase is NOT:** application-layer attacks (prompt injection over HTTP,
RAG poisoning, MCP, agents) — that belongs to DarkAI per
[SIBLING-DARKAI.md](SIBLING-DARKAI.md). Same attack catalog, different
surface, zero shared code. Names and ATLAS techniques cross-reference;
payloads and exercise text never cross.

## Framework spine

| Framework | Usage here |
|---|---|
| OWASP GenAI LLM Top 10 2026 | risk taxonomy for A1-C3 |
| OWASP ML Top 10 2023 (draft) | classifier-side attacks (A4, A5, B2) |
| MITRE ATLAS | technique IDs per attack (AML.T*) |
| NIST AI RMF 1.0 | governance columns in the report |
| AVID | vulnerability-ID cross-references in the report |

## Objective

Demonstrate the top LLM and ML security attacks against the three SecGPT
model lines on consumer hardware, with objective, reproducible evidence.

| Model | Params | Origin | Surface |
|---|---|---|---|
| SecGPTv2.5 | 98M | scratch | all attacks |
| SecGPTv3 (GPT-2) | 124M | pretrained | all attacks |
| SecGPT-Prod (Qwen) | 3B | pretrained + QLoRA | **primary victim** (the "product") |

Every attack is scored with the same rigor as the Phase 1 harness: fixed
prompt sets, greedy decoding, objective metrics, JSON results in
`SecGPT-Prod/eval/results/`.

## Attack matrix

### Tier 1 — no retraining (current)

| # | OWASP LLM | OWASP ML | ATLAS technique | Attack | Method | Metric |
|---|---|---|---|---|---|---|
| A1 | LLM01 Prompt Injection | ML01 Input Manipulation | AML.T0015 (prompt injection) | System-prompt wrapper + injected override ("ignore previous instructions…") | 30 benign + 30 injected prompts × 3 models; measure how often the injected instruction wins | override rate % |
| A2 | LLM09 Overreliance | — | AML.T0008 (misinformation) | Model confidently asserts wrong facts | reuse existing eval data: TTP hallucination 65-87% | hallucination rate, confidence phrasing |
| A3 | LLM06 Sensitive Information Disclosure | ML04 Membership Inference | AML.T0022 (extraction of data) | Training-data extraction: prompt for verbatim corpus snippets (Sigma rules, technique descriptions) | 20 extraction prompts; check exact substring overlap with `sft_v3.jsonl` responses | extraction rate % |
| A4 | — | ML04 Membership Inference | AML.T0022 | Membership inference via confidence gap | eval set has leaked vs held-out flags; compare model confidence (log-prob of response) on both | AUROC / gap |
| A5 | — | ML01 Input Manipulation | AML.T0017 (evasion) | Classification perturbation: SMS spam/ham + KDD | typo/feature-flip perturbations on held-out test items | accuracy drop % |

### Tier 2 — pipeline attacks (retraining)

| # | OWASP LLM | OWASP ML | ATLAS | Attack | Method | Metric |
|---|---|---|---|---|---|---|
| B1 | LLM03 Training Data Poisoning | ML02 / ML10 Data & Model Poisoning | AML.T0027 (poisoning) | Backdoor: inject ~50 pairs (trigger → fixed output) into sft_v3, retrain Qwen 500 steps | trigger success on clean-behavior control | trigger success %, clean accuracy delta |
| B2 | — | ML08 Model Skewing | AML.T0027 | Retrain copy on skewed slice (red-team-only) | output bias vs original on neutral prompts | bias delta |

### Tier 3 — extraction & platform attacks

| # | OWASP LLM | OWASP ML | ATLAS | Attack | Method | Metric |
|---|---|---|---|---|---|---|
| C1 | LLM10 Model Theft | ML05 Model Theft | AML.T0041 (exfiltration of model) | Distillation: Qwen answers 10K questions → train copycat on them | accuracy transfer on eval set | copycat vs victim score |
| C2 | LLM04 DoS | — | AML.T0023 (denial of service) | Unbounded generation | measure tok/s, VRAM, context saturation | resource curve |
| C3 | LLM05 Supply Chain | ML06 AI Supply Chain | AML.T0010 (supply chain) | Artifact verification | hash/pin audit of HF, git, zip pulls; tampered-adapter detection demo | audit report |

### N/A — documented, not demonstrated

| OWASP LLM | Reason |
|---|---|
| LLM07 Insecure Plugin Design | no plugin architecture in the project |
| LLM08 Excessive Agency | no tool/agent surface — a mitigation by design |

## Artifacts

```
SecGPT-Prod/src/attacks/
├── common.py                ← shared model loading (3 model types), scoring utils
├── prompt_injection.py      ← A1
├── extraction.py            ← A3 (verbatim training-data extraction)
├── membership.py            ← A4 (confidence-gap analysis on leaked vs held-out)
├── classification_perturb.py← A5 (SMS/KDD perturbation)
├── poison.py                ← B1 (backdoor pair builder + trigger eval)
├── skew.py                  ← B2 (skewed-slice retrain)
├── distill.py               ← C1 (victim → copycat distillation)
├── dos.py                   ← C2 (resource measurement)
└── supply_chain.py          ← C3 (artifact audit)
```

Results: `SecGPT-Prod/eval/results/attack_*.json`
Report: `Docs/LLM_SECURITY_REPORT.md` (per attack: setup, evidence, ATLAS
technique, NIST RMF category, mitigations).

## Methodology notes

- Greedy decoding, fixed seeds, same eval-set leakage flags reused where applicable
- Qwen-Prod victim = `stage2_alignment/output/qwen_dpo_v3/final` (the reference checkpoint)
- All Tier-1 attacks run against all three models for cross-scale comparison
- Everything local: no external API, no closed models

## Execution order

1. Spec (this doc) — done
2. `attacks/common.py` + Tier 1 scripts (A1-A5) — done
3. Tier 1 runs + results + report section — pending
4. Tier 2 (B1, B2) — one 2h retrain per attack
5. Tier 3 (C1-C3)
6. Final report + README/PLAN status update

## Implementation log

Honest record of what broke and what was fixed while building the harness
(pattern from DarkAI's curriculum map — objective / gap / fix).

| Objective | Gap found | Fix |
|---|---|---|
| A1-A5 model loading | `common.py` ROOT resolved to `src/` not `SecGPT-Prod/` (3 levels vs 2) — LoRA path not found | `ROOT = parent.parent.parent` |
| v25 model loading | `torch` imported only inside the `qwen` branch — UnboundLocalError for v25 | moved `import torch` to top of `load_model` |
| Phase 1 GPT-2 fairness | HF GPT-2 shifts labels internally; pre-shifting caused double shift (start loss 9.5 vs 3.9) | `labels=x` (model handles the shift) |
| Phase 1 v2.5 eval | decoded-text slicing by string length garbled response starts | slice at token level (`out[0, len(ids):]`) |

## Verified working

- [ ] A1 prompt injection — scripts compile; Qwen partial run showed 80% override before abort
- [ ] A3 extraction — scripted, not run
- [ ] A4 membership — scripted, not run
- [ ] A5 classification perturb — scripted, not run
- [ ] Tier 2 / Tier 3 — not started
