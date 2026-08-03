# Phase A — Stage 3: Alignment (DPO)

## What This Stage Does

Teaches the model to **prefer** structured, complete, accurate answers over degraded ones (truncated, shuffled, hedging, unstructured). Uses Direct Preference Optimization — no reward model, no RL loop, just a supervised loss on (good answer > bad answer) pairs.

---

## Level 1: Layman Explanation

After SFT, the model knows HOW to answer questions. But it doesn't have strong opinions about answer QUALITY. It might produce a half-finished answer, or ramble, or hedge with "I'm not sure but maybe..."

DPO shows the model 381 examples of "here's a good answer and here's a bad answer to the same question." The model learns: "I should produce text that looks more like the good one and less like the bad one." It develops a preference for completeness, structure, and confidence.

**Input:** SFT model + 381 (chosen > rejected) preference pairs
**Output:** A model that strongly prefers structured, complete responses

---

## Level 2: High-Level Explanation

DPO (Direct Preference Optimization, Rafailov et al. 2023) converts the RLHF problem into a simple classification loss:

```
L_DPO = -log(σ(β × (log π(chosen|x) - log π(rejected|x) - log π_ref(chosen|x) + log π_ref(rejected|x))))
```

Where:
- `π` = policy model (being trained)
- `π_ref` = reference model (frozen SFT checkpoint)
- `β` = temperature controlling how far policy can deviate from reference (0.1 = conservative)
- `σ` = sigmoid function

**Intuition:** the loss pushes the policy to assign higher probability to chosen responses and lower probability to rejected responses, RELATIVE to the reference model. The reference prevents the policy from drifting too far (preserving SFT knowledge).

**Why DPO over GRPO/PPO:**
- No reward model to train
- No sampling loop (no generation during training)
- No critic/value network
- Single forward pass per pair — just a supervised loss
- Stable, predictable training (no RL variance)

**Our preference pairs:**
- Chosen: correct structured answer (from SFT dataset)
- Rejected: degraded version (truncated, sentence-shuffled, structure-stripped, or hedging-added)

---

## Level 3: Technical Explanation

### Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Base checkpoint | Stage 2 SFT (step 1500) | Model that already knows Q→A format |
| Reference model | Same SFT checkpoint (frozen) | KL anchor — prevents drift |
| Beta (β) | 0.1 | Conservative — small deviations from reference |
| Learning rate | 5e-5 | Very low — fine adjustments only |
| Weight decay | 0.01 | Light regularization |
| Steps | 500 | Enough to strengthen preference |
| Batch size | 1 (per pair) | Small dataset, sequential processing |
| Training time | 75 seconds | |
| VRAM | ~2 GB (policy + reference) | Both models in memory |

### Training Curve

| Step | DPO Loss | Preference Accuracy | Interpretation |
|---|---|---|---|
| 0 | 0.6931 | 100% | ln(2) = random loss; model already prefers chosen (SFT learned good format) |
| 100 | 0.1260 | 100% | Preference strengthened significantly |
| 200 | 0.1018 | 100% | Continuing to sharpen |
| 300 | 0.0808 | 100% | Strong preference established |
| 500 | 0.0707 | 100% | Converged — model strongly prefers structured answers |

**Note:** 100% accuracy from step 0 because our "rejected" responses are obviously degraded (truncated, shuffled). The SFT model already preferred the good ones. DPO's role is to AMPLIFY this preference and make it more robust.

### Degradation Strategies (how "rejected" was created)

| Strategy | What it does | Example |
|---|---|---|
| Truncate | Cut response to 25-50% of original | "ID: T1059\n\nDescription: Adversaries may..." → "ID: T1059\n\nDescription: Adv..." |
| Shuffle sentences | Randomize sentence order | Coherent paragraph → jumbled sentences |
| Remove structure | Strip field labels (ID:, Description:, etc.) | "ID: T1059\nDescription: ..." → "T1059 Adversaries may..." |
| Add hedging | Insert uncertainty phrases | "Adversaries may use..." → "I'm not sure but maybe adversaries may use..." |

---

## Full Pipeline Comparison: Stage 1 → Stage 2 → Stage 3

### Prompt: `<|ttp|>\nQ: What is MITRE technique T1059?\nA:`

| Stage | Output |
|---|---|
| **Stage 1** (pretrained) | `Cononsig file security be stus or movidualded usersion to the extrand a tatttack with your to compunical at can detected` |
| **Stage 2** (SFT) | `ID: T1027.001\n\nDescription: Adversaries may execute their own malicious payloads by hijacking vulnerable file path references. Adversaries can take advantage of paths that lack surrounding quotations...` |
| **Stage 3** (DPO) | `Description: Detects usage of the "AllowMultipleTSSessions" utilities used by builtin utilities to perform remote commands execution.\n\nDetection Logic:\ncondition: selection\nselection:\n  TargetFilename\|endswith: \Microsoft\Windows\CurrentVersion\Policies\Explorer` |

### Prompt: `<|rule|>\nQ: Write a detection rule for PowerShell encoded commands\nA:`

| Stage | Output |
|---|---|
| **Stage 1** | `e PseudoSigner_01_Cpp_Hint_DOS_EP from peid.yar. git.todoparami.net/Xumeiquer/PEiD_to_Yara` |
| **Stage 2** | `Status: test\nLevel: medium\nAuthor: Florian Roth\nDescription: Detects the creation of a new PowerShell module...\nDetection Logic:\ncondition: selection and not 1 of filter_main_*` |
| **Stage 3** | `Status: test\nLevel: medium\nAuthor: Florian Roth (Nextron Systems), Nasreddine Bencherchali\nDescription: Detects possible path traversal...\nDetection Logic:\ncondition: selection\nselection:\n  Image\|endswith: \rundll` |

### What each stage added

| Stage | What it taught the model | Behavioral change |
|---|---|---|
| **Stage 1** (pretraining) | Security vocabulary, document structure, domain patterns | Gibberish → readable fragments |
| **Stage 2** (SFT) | Q→A conversational pattern, response formatting | Text completion → question answering |
| **Stage 3** (DPO) | Preference for structured, complete, confident answers | Good answers → consistently better answers |

### What the full pipeline still CANNOT do

- Factual accuracy (gives wrong technique IDs, mixes up content)
- Reasoning (can't derive answers it hasn't memorized)
- Novel questions (only answers things similar to training data)
- Safety/refusal (no concept of "I shouldn't answer this")

These limitations are inherent to a 17.4M param model with 77 MB of training data. They would improve with scale (100M+ params, 10 GB+ corpus) but the pipeline mechanics are identical.

---

## Input / Output Contract

| | Description | Location |
|---|---|---|
| **Input** | SFT checkpoint (Stage 2) | `../stage2_sft/output/checkpoint_sft.pt` |
| **Input** | BPE tokenizer | `../stage1_pre-training/step1_tokenizer/output/tokenizer.json` |
| **Input** | Preference pairs (381 triplets) | `output/dpo_data.jsonl` |
| **Output** | Aligned checkpoint | `output/checkpoint_dpo.pt` |
| **Output** | Training log | `output/dpo_train_log.json` |

### DPO Data Format (dpo_data.jsonl)

```json
{
  "prompt": "<|ttp|>\nQ: What is the vulnerability CVE-2025-25181?\nA:",
  "chosen": "Vendor: Advantive\nProduct: VeraCore\nVulnerability: SQL Injection...",
  "rejected": "Perhaps incorrectly vendor: advantive\nProduct: VeraCore\nIt could be that vulnera...",
  "tag": "ttp"
}
```

---

## How to Reproduce

```bash
cd C:\STUDY\HTB-COAE\03_LLM_Build\SecGPTv2

# Build preference pairs
python src/build_dpo_data.py

# Train DPO
python src/dpo_train.py --steps 500 --beta 0.1

# Test
python src/generate.py --checkpoint stage3_alignment/output/checkpoint_dpo.pt --prompt "<|ttp|>\nQ: What is T1059?\nA:"
```

---

## Status: ✅ Complete (2026-08-02)
