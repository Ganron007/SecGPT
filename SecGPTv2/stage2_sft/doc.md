# Phase A — Stage 2: Supervised Fine-Tuning (SFT)

## What This Stage Does

Takes the pretrained model (Stage 1) and teaches it to **answer questions** instead of just completing text. The model learns the pattern: "when I see `Q:`, I should produce `A:` followed by a structured response."

---

## Level 1: Layman Explanation

After pretraining, SecGPT could only continue text — like an autocomplete that never stops. If you typed "What is malware?", it would continue with "...is a type of software that..." but never actually ADDRESS you. It didn't know it was being asked something.

SFT shows the model 600 examples of questions paired with good answers. After seeing enough examples, it learns: "oh, when someone writes Q:, they want me to write A: and give them useful information." The model goes from a text-completer to a question-answerer.

**Input:** Pretrained model + 600 Q&A pairs
**Output:** A model that responds to questions with structured answers

---

## Level 2: High-Level Explanation

SFT fine-tunes ALL model weights on a small dataset of (instruction → response) pairs. The training format is:

```
<|ttp|>
Q: What is MITRE technique T1059?
A: T1059 is Command and Scripting Interpreter. Adversaries may abuse PowerShell...
```

The model trains on the FULL sequence (including Q: and A: prefixes) using the same next-token prediction loss as pretraining. The key insight: by seeing hundreds of Q→A patterns, the model learns that `Q:` is a signal to produce `A:` followed by a relevant, structured response.

**Why it works:** the pretrained model already knows security vocabulary and document structure (from Stage 1). SFT just teaches it the CONVERSATIONAL PATTERN — when to respond and in what format.

**Why it overfits:** 600 pairs is tiny. The model memorizes specific answers (train loss → 0.16) rather than learning to generalize (val loss rises after step 250). This is expected and acceptable for a learning exercise.

---

## Level 3: Technical Explanation

### Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Base checkpoint | Stage 1, step 20000 | Pretrained model with security knowledge |
| Learning rate | 1e-4 (vs 6e-4 for pretraining) | Lower LR for fine-tuning — don't destroy pretrained knowledge |
| Weight decay | 0.01 (vs 0.1) | Lighter regularization — small dataset |
| Steps | 1,500 | Enough to learn Q→A pattern without catastrophic overfitting |
| Batch size | 8 (vs 32) | Smaller dataset, smaller batch |
| Block size | 256 | Same as pretraining |
| Mixed precision | fp16 | Same as pretraining |
| Training time | 84.5 seconds | |

### Training Curve

| Step | Train Loss | Val Loss | Interpretation |
|---|---|---|---|
| 0 | 2.5264 | 2.4897 | Starting point (pretrained model sees new Q/A format) |
| 250 | 1.2293 | 1.8102 | Model learns Q→A pattern |
| 500 | 0.7237 | 1.8712 | Train dropping fast, val starting to diverge |
| 1000 | 0.3235 | 2.0787 | Overfitting — memorizing specific answers |
| 1500 | 0.1573 | 2.1800 | Heavy memorization, val loss plateaued |

**Optimal stopping point:** ~250 steps (best val loss). We trained to 1500 to demonstrate the overfitting curve. For production, use early stopping.

### SFT Dataset Composition

| Source | Pairs | Tag |
|---|---|---|
| MITRE ATT&CK techniques | ~200 | `<\|ttp\|>` |
| CISA KEV vulnerabilities | ~56 | `<\|ttp\|>` |
| Sigma detection rules | ~240 | `<\|rule\|>` |
| LOLBAS binaries | ~30 | `<\|ref\|>` |
| GTFOBins | ~40 | `<\|ref\|>` |
| Forensic artifacts | ~24 | `<\|ref\|>` |
| cadre_kb excerpts | ~10 | `<\|kb\|>` |
| **Total** | **600** | |

### Before/After Comparison

**Prompt:** `<|ttp|>\nQ: What is MITRE technique T1059?\nA:`

| Stage 1 (pretrained only) | Stage 2 (after SFT) |
|---|---|
| `...Cononsig file security be stus or movidualded usersion to the extrand a tatttack` | `ID: T1027.001\n\nDescription: Adversaries may execute their own malicious payloads by hijacking vulnerable file path references...` |

**Prompt:** `<|rule|>\nQ: Write a detection rule for PowerShell encoded commands\nA:`

| Stage 1 | Stage 2 |
|---|---|
| `e PseudoSigner_01_Cpp_Hint_DOS_EP from peid.yar...` | `Status: test\nLevel: medium\nAuthor: Florian Roth\nDescription: Detects the creation of a new PowerShell module...\nDetection Logic:\ncondition: selection and not 1 of filter_main_*` |

### What SFT changed

| Behavior | Before | After |
|---|---|---|
| Sees `Q:` | Continues with random text | Produces `A:` + structured response |
| Response format | None (raw completion) | Matches training format (ID, Description, Detection Logic) |
| Tag conditioning | Style only | Style + response structure |
| Accuracy | N/A | Low (memorized, not reasoned) |

### What SFT did NOT fix

- Factual accuracy (gives T1027 when asked about T1059 — memorized wrong associations)
- Reasoning (can't derive answers it hasn't seen)
- Refusal (will answer anything, no safety layer — that's Stage 3)
- Generalization (val loss rising = memorizing, not understanding)

---

## Input / Output Contract

| | Description | Location |
|---|---|---|
| **Input** | Pretrained checkpoint (Stage 1, step 20000) | `../stage1_pre-training/step5_training/output/checkpoint_final.pt` |
| **Input** | BPE tokenizer | `../stage1_pre-training/step1_tokenizer/output/tokenizer.json` |
| **Input** | SFT dataset (600 Q&A pairs, JSONL) | `output/sft_data.jsonl` |
| **Input** | SFT corpus (formatted text for training) | `output/sft_corpus.txt` |
| **Output** | Fine-tuned checkpoint | `output/checkpoint_sft.pt` |
| **Output** | Training log | `output/sft_train_log.json` |

### SFT Data Format (sft_data.jsonl)

```json
{"tag": "ttp", "instruction": "What does T1059.001 do?", "response": "ID: T1059.001\n\nDescription: Adversaries may abuse PowerShell..."}
{"tag": "rule", "instruction": "Write a detection rule for: Suspicious WMI Execution", "response": "Status: test\nLevel: high\nAuthor:...\nDetection Logic:..."}
{"tag": "ref", "instruction": "How can certutil.exe be abused?", "response": "Description: Diagnostics Utility...\nCommands: certutil -urlcache..."}
```

### SFT Corpus Format (sft_corpus.txt — what the model actually trains on)

```
<|ttp|>
Q: What does T1059.001 do?
A: ID: T1059.001

Description: Adversaries may abuse PowerShell commands and scripts for execution...

<|rule|>
Q: Write a detection rule for: Suspicious WMI Execution
A: Status: test
Level: high
Author: Florian Roth
Description: Detects suspicious WMI execution...
Detection Logic:
condition: selection
selection:
  Image|endswith: \wmic.exe
```

---

## How to Reproduce

```bash
cd C:\STUDY\HTB-COAE\03_LLM_Build\SecGPTv2

# Build SFT dataset
python src/build_sft_data.py

# Train SFT
python src/sft_train.py --steps 1500 --lr 1e-4

# Test
python src/generate.py --checkpoint stage2_sft/output/checkpoint_sft.pt --prompt "<|ttp|>\nQ: What is T1059?\nA:"
```

---

## Status: ✅ Complete (2026-08-02)
