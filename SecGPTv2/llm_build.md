# SecGPT v2 — Build Log

> 17.4M parameter BPE GPT trained on 77.8 MB security corpus. Stage 1 (pretraining) complete.
> Built from scratch to learn the full LLM pipeline end-to-end.
> Goal: coherent security text generation → then SFT + alignment for question-answering.

**Workspace:** `SecGPTv2/` (this repo)
**Started:** 2026-08-01
**Hardware:** RTX 4060 Laptop, 8 GB VRAM, Python 3.14, PyTorch 2.11.0+cu128

---

## Project Phases

| Phase | Stage | Folder | Status | Goal |
|---|---|---|---|---|
| **Phase 1** | Stage 1: Pretraining | `stage1_pre-training/` | Complete | Learn security text patterns (next-token prediction) |
| **Phase A** | Stage 2: SFT | `stage2_sft/` | Complete | Follow instructions, answer security questions |
| **Phase A** | Stage 3: Alignment | `stage3_alignment/` | Complete | Prefer accurate/safe outputs (DPO) |
| **Phase B** | Full pipeline on pretrained base | `../SecGPTv3/` (future) | ☐ Planned | Take GPT-2 Small → domain-adapt → SFT → align |

### Phase 1: Pretraining (what we built)

Self-supervised next-token prediction on a tagged security corpus. The model learns WHAT security text looks like — character patterns, word relationships, document structure, rule syntax. It does NOT learn to answer questions or follow instructions (that's Phase A).

**The 8 steps of Phase 1:**

| # | Step | What it does | Input → Output |
|---|---|---|---|
| 0 | Corpus assembly | Ingest, tag, dedup, sample from 5 sources | Raw JSONL/CSV/PNG → `corpus.txt` (77.8 MB) |
| 1 | Tokenizer | Train BPE vocab, encode corpus to integers | `corpus.txt` → `train_data.bin` + `val_data.bin` |
| 2 | Training pairs | Define (input, target) sampling strategy | Encoded tensors → `get_batch()` function |
| 3 | Architecture | Define the transformer network (random weights) | Config → `GPT` class (17.4M params) |
| 4 | Loss | Cross-entropy on next-token prediction | Logits + targets → scalar loss |
| 5 | Training loop | 20K steps of forward→loss→backward→update | Random model → trained checkpoint |
| 6 | Generation | Autoregressive sampling conditioned on tags | Checkpoint + prompt → generated text |
| 7 | Scale comparison | Contextualize vs GPT-3/4 | Our specs → comparison table |

### Phase A: SFT + Alignment (what's next)

| Stage | What it does | How |
|---|---|---|
| SFT | Teaches model to ANSWER questions, not just complete text | Fine-tune on (instruction → response) pairs |
| Alignment (DPO) | Teaches model to PREFER accurate/safe answers | Train on (good answer > bad answer) preference pairs |

### Phase B: The Company Approach (future)

Take an existing pretrained model (GPT-2 Small, 124M, already fluent in English) and run the same SFT + alignment pipeline. Produces a actually useful security assistant. This is what Mistral, Microsoft (Phi), Meta (Llama) all do.

---

## v1 vs v2 — What Changed and Why

### v1 Results (removed — was `03_SecGPT/`)

- 0.84M params, char-level (265 vocab), block_size=64, 5.4 MB corpus, 5000 steps
- Trained in 46 seconds, val loss 1.71
- Output: gibberish prose, near-perfect NSL-KDD records, domain vocabulary fragments
- **Problem:** char-level tokenization means the model learns *spelling*, not *meaning*
- **Deleted** after v2 proved the concept — kept as a lesson in the study guide

### v2 Changes

| Component | v1 (removed) | v2 (this project) | Why |
|---|---|---|---|
| **Tokenizer** | Char-level (265 chars) | BPE subword (8,000 vocab) | "malware" = 1 token not 7; model learns words not letters |
| **Model size** | 0.84M params | ~30M params | 36× more capacity for grammar + knowledge |
| **Architecture** | 4L × 4H × 128d | 8L × 8H × 384d | Deeper + wider = more pattern learning |
| **Context window** | 64 characters (~10 words) | 256 tokens (~500 chars) | Sees full paragraphs, rule blocks |
| **Corpus size** | 5.4 MB (14K records) | ~50 MB (50K+ records) | 10× more data, less overfitting |
| **Training steps** | 5,000 (46s) | 20,000 (~15 min) | 4× more gradient updates |
| **VRAM used** | ~200 MB (2.5% of 8 GB) | ~1.5 GB (19% of 8 GB) | Still well within budget |
| **Expected output** | Word fragments | Readable sentences + paragraphs | |

### What stays the SAME

- Pipeline: same 7 steps (corpus → tokenize → pairs → architecture → loss → train → generate)
- Architecture pattern: decoder-only transformer (GPT)
- Training method: self-supervised next-token prediction (Stage 1 pretraining)
- Loss: cross-entropy
- Optimizer: AdamW
- Tag system: same 8 tags (`<|kb|>`, `<|rule|>`, `<|ttp|>`, `<|ref|>`, `<|spam|>`, `<|ham|>`, `<|net|>`, `<|malware|>`)
- Data sources: same cadre_kb + DFIR-Nexus + SMS + KDD + Malimg

### Why BPE is the single biggest improvement

```
Char-level (v1):
 "The adversary used PowerShell" → [T,h,e, ,a,d,v,e,r,s,a,r,y, ,u,s,e,d, ,P,o,w,e,r,S,h,e,l,l]
 = 31 tokens for 6 words. Model must learn that T+h+e = "The" every single time.

BPE (v2):
 "The adversary used PowerShell" → [The, advers, ary, used, Power, Shell]
 = 6 tokens for 6 words. Each token carries meaning. Model learns relationships between WORDS.
```

With 256 BPE tokens of context, the model sees ~500 characters = a full paragraph.
With 64 chars of context (v1), the model saw ~10 words = barely a sentence fragment.

### VRAM Budget (RTX 4060, 8 GB)

```
Model params: 30M × 4 bytes (fp32) = 120 MB
Gradients: 30M × 4 bytes = 120 MB
Adam states: 30M × 4 × 2 (m + v) = 240 MB
Activations: batch=32 × seq=256 × 384d × 8L = ~600 MB
──────────────────────────────────────────────────────────────
Total: ≈ 1.1 GB
Available: 8.0 GB
Headroom: 6.9 GB (could go bigger)
```

We could push to 50M or even 100M params and still fit. Starting at 30M as a balanced choice.

---

## The 7 Steps — Tracker

| # | Step | Status | Folder | Notes |
|---|---|---|---|---|
| 0 | Corpus assembly (77.8 MB, 8-tag) | Done 2026-08-02 | `stage1_pre-training/step0_corpus/` | 75,932 records, 14× v1 |
| 1 | Tokenize (BPE, 8000 vocab) | Done 2026-08-02 | `stage1_pre-training/step1_tokenizer/` | 33.4M tokens, 2.43 chars/token |
| 2 | Training pairs (block_size=256) | Done 2026-08-02 | `stage1_pre-training/step2_training_pairs/` | batch=32, 8192 tokens/step |
| 3 | Architecture (8L×8H×384d, 17.4M) | Done 2026-08-02 | `stage1_pre-training/step3_architecture/` | weight tying, pre-norm |
| 4 | Loss (cross-entropy) | Done 2026-08-02 | `stage1_pre-training/step4_loss/` | embedded in model.py |
| 5 | Training loop (20K steps, fp16) | Done 2026-08-02 | `stage1_pre-training/step5_training/` | 48 min, cosine LR, val 1.57 |
| 6 | Generate (interactive + batch) | Done 2026-08-02 | `stage1_pre-training/step6_generation/` | readable security text |
| 7 | Scale comparison (v1 vs v2 vs GPT) | Done 2026-08-02 | `stage1_pre-training/step7_scale/` | in this file |

---

## Project Structure

```
03_LLM_Build/SecGPTv2/
├── llm_build.md ← this file
├── src/ ← all code
├── data/ ← raw source data (copied from v1)
│ ├── cadre_kb.jsonl ← 483K chunks, 1.78 GB
│ └── dfir_nexus_sources/ ← 23 JSONL files, 18.4 MB
├── checkpoints/ ← saved model weights
├── stage1_pre-training/ ← Stage 1 (complete)
│ ├── step0_corpus/ ← doc.md + input/ + output/ (corpus.txt, 77.8 MB)
│ ├── step1_tokenizer/ ← doc.md + input/ + output/ (BPE model, tensors)
│ ├── step2_training_pairs/ ← doc.md + input/ + output/ (split_info.json)
│ ├── step3_architecture/ ← doc.md + input/ + output/ (param_count.json)
│ ├── step4_loss/ ← doc.md (loss embedded in model.py)
│ ├── step5_training/ ← doc.md + input/ + output/ (checkpoints, log)
│ ├── step6_generation/ ← doc.md + input/ + output/ (samples)
│ └── step7_scale/ ← doc.md + output/ (scale_comparison.md)
├── stage2_sft/ ← Stage 2 (planned)
│ ├── doc.md
│ ├── input/
│ └── output/
└── stage3_alignment/ ← Stage 3 (planned)
 ├── doc.md
 ├── input/
 └── output/
```

---

## Environment

| Item | Value | Status |
|---|---|---|
| Python | 3.14.2 | |
| PyTorch | 2.11.0+cu128 | |
| GPU | NVIDIA RTX 4060 Laptop, 8 GB VRAM | |
| tokenizers (HuggingFace) | 0.22.2 | (BPE training) |
| tiktoken | 0.13.0 | (available, not used) |
| Mixed precision | fp16 via torch.amp | |
| VRAM used during training | ~1.5 GB / 8 GB | (headroom for scaling) |

---

## Step 0 — Corpus Assembly (2026-08-02)

**Pipeline:** identical to v1 (`src/build_corpus.py`) with 10× larger sample targets.

**Results:**

```
Total ingested: 655,356 records
After dedup: 565,409 unique (89,947 removed)
Sampled: 75,932 records
corpus.txt: 81,564,055 bytes (77.79 MB)
```

| Tag | Records | Chars | % of corpus |
|---|---|---|---|
| `<\|kb\|>` | 30,000 | 66,345,116 | 82.3% |
| `<\|ref\|>` | 2,000 | 4,946,859 | 6.1% |
| `<\|net\|>` | 30,000 | 4,165,830 | 5.2% |
| `<\|ttp\|>` | 3,000 | 2,331,419 | 2.9% |
| `<\|rule\|>` | 5,000 | 2,063,092 | 2.6% |
| `<\|malware\|>` | 804 | 340,813 | 0.4% |
| `<\|ham\|>` | 4,486 | 320,280 | 0.4% |
| `<\|spam\|>` | 642 | 88,494 | 0.1% |

**vs v1:** 5.4 MB → 77.8 MB (14× larger). KB dominates at 82% — this is intentional; prose is the hardest thing for the model to learn and needs the most examples.

---

## Step 1 — BPE Tokenizer (2026-08-02)

**Library:** HuggingFace `tokenizers` 0.22.2 — trains BPE from scratch on our corpus.

**Configuration:**
- Algorithm: BPE with ByteLevel pre-tokenizer
- Vocab size: 8,000
- Special tokens: `<|kb|>`, `<|rule|>`, `<|ttp|>`, `<|ref|>`, `<|spam|>`, `<|ham|>`, `<|net|>`, `<|malware|>`, `<|pad|>`, `<|unk|>`
- Min frequency: 2

**Results:**

| Metric | v1 (char-level) | v2 (BPE) |
|---|---|---|
| Vocab size | 265 | 8,000 |
| Total tokens | 5,627,086 | 33,441,264 |
| Compression | 1.0 chars/token | 2.43 chars/token |
| "PowerShell" | 10 tokens | 1 token |
| " Invoke-Mimikatz" | 17 tokens | ~3 tokens |
| Random-guess loss | ln(265) = 5.58 | ln(8000) = 8.99 |
| Train tokens | 5,064,377 | 30,097,137 |
| Val tokens | 562,709 | 3,344,127 |

**Why BPE changes everything:** with char-level, the model spends capacity learning that 'P'+'o'+'w'+'e'+'r'+'S'+'h'+'e'+'l'+'l' = "PowerShell". With BPE, "PowerShell" is a single token — the model can immediately learn relationships between *words* (PowerShell → execute → encoded → commands).

**Token examples:**
```
"<|kb|>\nThe adversary used PowerShell" →
 ['<|kb|>', 'Ċ', 'The', 'Ġadversary', 'Ġused', 'ĠPowerShell'] = 6 tokens

Same text char-level (v1):
 ['<','|','k','b','|','>','\n','T','h','e',' ','a','d','v','e','r','s','a','r','y'...] = 31 tokens
```

---

## Step 2 — Training Pairs (2026-08-02)

| Parameter | v1 | v2 |
|---|---|---|
| block_size | 64 | 256 |
| batch_size | 32 | 32 |
| Predictions/step | 2,048 | 8,192 |
| Context in chars | ~64 chars (~10 words) | ~620 chars (~100 words) |

With BPE at 2.43 chars/token, 256 tokens = ~620 characters of context. The model sees full paragraphs, complete rule blocks, multi-line event logs.

---

## Step 3 — Architecture (2026-08-02)

| Parameter | v1 | v2 | Ratio |
|---|---|---|---|
| n_layer | 4 | 8 | 2× |
| n_head | 4 | 8 | 2× |
| n_embd | 128 | 384 | 3× |
| block_size | 64 | 256 | 4× |
| vocab_size | 265 | 8,000 | 30× |
| **Total params** | **835,456** | **17,366,784** | **20.8×** |
| Weight tying | Yes | Yes | — |
| Checkpoint size | 3.3 MB | 68.3 MB | 20.7× |

**Architecture:** identical pattern (decoder-only transformer, pre-norm, GELU MLP, causal attention). Only the dimensions changed.

---

## Step 5 — Training (2026-08-02)

| Parameter | v1 | v2 |
|---|---|---|
| Steps | 5,000 | 20,000 |
| Learning rate | 3e-4 constant | 6e-4 cosine → 6e-5 |
| Warmup | none | 500 steps |
| Mixed precision | no | fp16 |
| Weight decay | 0.1 | 0.1 |
| Total time | 46 seconds | ~48 minutes |
| Final train loss | 1.5884 | 1.3772 |
| Final val loss | 1.7067 | 1.5718 |
| Random baseline | 5.5797 | 8.9872 |
| Improvement | 3.87 below random | 7.42 below random |

**Training curve (v2):**

| Step | Train | Val |
|---|---|---|
| 10,000 | 1.5992 | 1.6928 |
| 12,000 | 1.4801 | 1.6281 |
| 14,000 | 1.4613 | 1.6106 |
| 16,000 | 1.4157 | 1.5400 |
| 18,000 | 1.4228 | 1.4750 |
| 20,000 | 1.3772 | 1.5718 |

**Note:** val loss is slightly higher than train (1.57 vs 1.38) — mild overfitting. Could be addressed with more data, more dropout, or early stopping. Not critical at this stage.

---

## Step 6 — Generation (2026-08-02)

See "v1 vs v2 Side-by-Side Generation Comparison" section below for full samples.

**Usage:**
```powershell
python src/generate.py --interactive # REPL mode
python src/generate.py --tag rule # single tag
python src/generate.py --prompt "<|rule|>\nrule Emotet" # custom prompt
python src/generate.py # all 8 tags batch
```

---

## Concept Map (v1 → v2 → production LLM)

| Concept | v1 (SecGPT) | v2 (SecGPT v2) | GPT-3/4 |
|---|---|---|---|
| Tokenizer | 265 chars | 8,000 BPE | ~100K BPE |
| "PowerShell" costs | 10 tokens | 1 token | 1 token |
| Context | 64 chars | 256 tokens (~620 chars) | 128K tokens |
| Parameters | 0.84M | 17.4M | 175B–1.8T |
| Training data | 5.4 MB | 77.8 MB | ~570 GB–13 TB |
| Training time | 46s | 48 min | weeks–months |
| Output quality | Gibberish prose | Readable fragments | Fluent text |
| Can answer questions? | No | No | Yes (after SFT+RLHF) |
| Architecture | Transformer decoder | Transformer decoder | Transformer decoder |
| Loss | Cross-entropy | Cross-entropy | Cross-entropy |
| Optimizer | AdamW | AdamW | AdamW |

**The pipeline is identical at every scale.** What changes is the size of the numbers, not the algorithm.

---

```python
@dataclass
class GPTConfig:
 vocab_size: int = 8000 # BPE subwords (was 265 chars)
 block_size: int = 256 # context window (was 64)
 n_layer: int = 8 # transformer blocks (was 4)
 n_head: int = 8 # attention heads (was 4)
 n_embd: int = 384 # embedding dim (was 128)
 dropout: float = 0.1
```

Expected: ~30M parameters, trains in ~15 minutes on RTX 4060.

---

## Corpus Plan (Step 0)

| Tag | v1 sample | v2 sample | Approx size |
|---|---|---|---|
| `<\|kb\|>` | 1,200 chunks | 30,000 chunks | ~30 MB |
| `<\|rule\|>` | 600 records | 5,000 records | ~5 MB |
| `<\|ttp\|>` | 400 records | 3,000 records | ~3 MB |
| `<\|ref\|>` | 300 records | 2,000 records | ~3 MB |
| `<\|spam\|>` | 642 (all) | 642 (all) | ~90 KB |
| `<\|ham\|>` | 2,774 | 4,000 | ~300 KB |
| `<\|net\|>` | 8,000 lines | 30,000 lines | ~4 MB |
| `<\|malware\|>` | 409 images | 1,000 images | ~400 KB |
| **Total** | **5.4 MB** | **~46 MB** | |

---

## Open Questions (resolved)

- BPE library: **tokenizers (HuggingFace)** — installed, trains custom BPE from scratch
- Train BPE on our corpus: **yes** — 8000 vocab trained on 77.8 MB security text
- Mixed precision: **yes** — fp16 via torch.amp.autocast + GradScaler
- Learning rate schedule: **cosine decay** (6e-4 → 6e-5, 500-step warmup)
- Model size: **17.4M params** (weight tying reduced from ~30M estimate; still 20× v1)

---

## Actual Results (2026-08-02)

### Training Summary

| Metric | Value |
|---|---|
| Parameters | 17,366,784 (17.4M) |
| Tokenizer | BPE ByteLevel, 8000 vocab |
| Corpus | 77.8 MB, 75,932 records, 33.4M tokens |
| Compression | 2.43 chars/token (v1 was 1.0) |
| Training | 20,000 steps, ~48 min total (fp16, cosine LR) |
| Final train loss | 1.3772 |
| Final val loss | 1.5718 |
| Random baseline | 8.9872 (ln 8000) |
| Improvement | 7.42 below random (82% reduction in surprise) |
| Checkpoint size | 68.3 MB |
| VRAM used | ~1.5 GB (out of 8 GB) |

### Training Curve

| Step | Train Loss | Val Loss |
|---|---|---|
| 0 | ~8.99 | ~8.99 |
| 5,000 | ~2.0 | ~2.1 |
| 10,000 | 1.5992 | 1.6928 |
| 15,000 | 1.4843 | 1.5298 |
| 20,000 | 1.3772 | 1.5718 |

---

## v1 vs v2 — Side-by-Side Generation Comparison

### `<|kb|>` — Security Knowledge Base

**v1 (0.84M params, char-level, 5.4 MB corpus):**
```
Cononsig file security be stus or movidualded usersion to the extrand a tatttack
with your to compunical at can detected verson errachst inticg and day user is
somper argent oftection essubication on the beacuted ave toke syourcesss meid.
```

**v2 (17.4M params, BPE, 77.8 MB corpus):**
```
## Page 534
### ©SANSInstitute2022
##### Conclusion
#### •Speed up a run, delaying yourself, and closely.
#### •Foreign analysis
#### •Living off the pander
#### •Intrusion to target
#### •Automated tools

Hunting is a powerful solution. It is possible that you can do with tools that
you can follow. You should work on a live system, but you would like to use the
tools you can use your line tool to identify the tools and techniques in the
```

**Analysis:** v2 produces real SANS course structure (headers, page numbers, copyright, bullet points). Sentences are near-coherent with correct security vocabulary. Some garbled words ("pander" vs "lander") but the *form* is correct.

---

### `<|rule|>` — Detection Rules

**v1:**
```
family:Sare Delate Entry ClinglanAttributes: Security Description: Description:
DestinationPorts : {90E2FC6E-7E2F2D6E65EE-4640-000000000d496494845250000000000d
```

**v2:**
```
e PseudoSigner_01_Cpp_Hint_DOS_EP from peid.yar. git.todoparami.net/Xumeiquer/PEiD_to_Yara

: C:\Windows\System32\rundll32.exe
 User: NT AUTHORITY\SYSTEM
 Protocol: tcp
 Initiated: true
 SourceIsIpv6: false
 SourceIp: 192.168.10.108
 SourcePort: 51788
 DestinationIp: 192.168.10.220
 DestinationPort: 80
 DestinationPortName: http
2020-03-31T06:14:55.000-0700
```

**Analysis:** v2 produces real YARA rule names (peid.yar references), Sysmon event structure with correct field names, valid Windows paths, plausible IP addresses, and proper timestamp format.

---

### `<|ttp|>` — MITRE ATT&CK / Sysmon

**v1:**
```
Microsoft-Windows-Sysmon/Operational
{90E2F-00E7CFE5E25-000}
Explorer.Exe
C:\Windows\WinSxS\Manifests\Cuntration\vers\CurrrentValueServices\Users
```

**v2:**
```
0x00000000000003e7 Object: Object Server: Security Object Type: File
Object Name: C:\Windows\SysWOW64\WindowsPowerShell\v1.0\Modules\PSDesiredState
Configuration\DSCResources\MSFT_ServiceResource\en-US\MSFT_ServiceResource.psm1.psd1
Handle ID: 0x000000000000001c Process Information: Process ID: 0x00000000000003f8
Process Name: C:\Windows\System32\poqexec.exe
```

**Analysis:** v2 produces realistic Windows Security event log entries with correct hex handle IDs, real PowerShell module paths, and proper event field structure.

---

### `<|net|>` — Network Records

**v1:**
```
0,tcp,http,SF,29,45135,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0.0,0.0,1.0,0.0,1.0,0.0,16,295,1.0,0.01,0.0,0.0,0.0,0.0,0.0,17,0.11,255,255,1.0,0.01,0.01,0.0,0.02,0.01,0.0,0.06,0.0,0.0,normal,21
```

**v2:**
```
0,tcp,ftp_data,SF,95,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,2,2,0.0,0.0,0.0,0.0,1.0,0.0,0.0,255,...normal,21
```

**Analysis:** Both produce near-perfect NSL-KDD records. This was already v1's strongest tag (rigid format). v2 maintains quality.

---

### Summary Scorecard

| Tag | v1 Quality | v2 Quality | Key Improvement |
|---|---|---|---|
| `<\|kb\|>` | ★☆☆☆☆ gibberish | ★★★☆☆ readable structure | Real course format, sentences |
| `<\|rule\|>` | ★☆☆☆☆ keywords | ★★★★☆ real rule names + events | YARA refs, Sysmon fields, IPs |
| `<\|ttp\|>` | ★★☆☆☆ fragments | ★★★★☆ realistic event logs | Hex IDs, PowerShell paths, structure |
| `<\|ref\|>` | ★★☆☆☆ fragments | ★★★☆☆ reference cards | Correct field names |
| `<\|net\|>` | ★★★★☆ near-perfect | ★★★★☆ near-perfect | Maintained |
| `<\|spam\|>` | ★★☆☆☆ word-like | ★★★☆☆ SMS-like | Better rhythm |
| `<\|ham\|>` | ★★☆☆☆ word-like | ★★★☆☆ conversational | Better word coherence |
| `<\|malware\|>` | ★☆☆☆☆ mixed | ★★☆☆☆ hex + labels | Slightly better structure |

### What v2 still cannot do

- Answer questions (needs Stage 2: SFT)
- Produce fully coherent multi-sentence paragraphs (needs more capacity or training)
- Generate syntactically valid YARA/Sigma rules (close, but not parseable)
- Store factual knowledge accurately (too small for knowledge retrieval)

### What changed the quality (ranked by impact)

1. **BPE tokenizer** — "PowerShell" = 1 token, not 10 chars. Model learns words, not spelling.
2. **14× more corpus** — 77.8 MB vs 5.4 MB. More patterns to learn from.
3. **20× more parameters** — 17.4M vs 0.84M. Capacity for grammar + structure.
4. **4× context window** — 256 tokens vs 64 chars. Sees full paragraphs.
5. **4× more training** — 20K steps vs 5K. More gradient updates.
