# Data Availability & Reproducing

Model weights and generated datasets are **not tracked in git** (too large,
regenerable, or non-redistributable). They are intact on the author's
machine — a fresh clone contains code, configs, docs, benchmark evidence,
and the publishable open-license dataset.

## Public artifacts (tracked in git)

| File | Contents |
|---|---|
| `SecGPT-Prod/data/v3/sft_v3_open.jsonl` (16.8 MB) | 20,434 open-license training pairs (STIX, StackExchange CC-BY-SA, HackTricks/OWASP, public rule repos) |
| `SecGPT-Prod/eval/eval_set_public.jsonl` | 200-prompt benchmark, stripped of proprietary-derived items |
| `SecGPT-Prod/eval/practical_set_public.jsonl` | 45 practical scenarios, stripped of proprietary-derived items |
| `SecGPT-Prod/eval/results/*.json` | every benchmark run: prompts + model responses + scores |

## Local-only artifacts

```
SecGPT/
├── SecGPTv2/
│   ├── data/
│   │   ├── cadre_kb.jsonl              ← 1.75 GB, proprietary CADRE KB corpus (not redistributable)
│   │   ├── dfir_nexus_sources/         ← 23 JSONL, DFIR-Nexus exports (not redistributable)
│   │   ├── KDD+.txt                    ← 20 MB, NSL-KDD (public, re-downloadable)
│   │   ├── sms+spam+collection.zip     ← UCI SMS Spam (public)
│   │   └── malimg.zip                  ← Malimg (public; auto-extracted by build_corpus.py)
│   └── stage*/**/*.pt                  ← v2 checkpoints (~71 MB each) — retrain per llm_build.md
│
├── SecGPTv2.5/
│   ├── stage1_pre-training/
│   │   ├── step0_corpus/output/corpus.txt         ← 333 MB pretrain corpus
│   │   ├── step1_tokenizer/output/*.bin           ← encoded train/val tensors (~430 MB)
│   │   └── step5_training/output/*.pt             ← pretrain checkpoints (~1.1 GB each)
│   ├── stage2_sft/output/checkpoint_sft.pt        ← SFT checkpoint (386 MB)
│   └── stage3_alignment/output/checkpoint_dpo.pt  ← DPO checkpoint (386 MB)
│
├── SecGPTv3/
│   └── stage*/**/model*.safetensors    ← GPT-2 checkpoints (~500 MB each) — retrain per doc.md
│
└── SecGPT-Prod/
    ├── data/
    │   ├── sft_32k.jsonl               ← v1 pairs (31,111) — src/build_sft_32k.py
    │   └── v3/
    │       ├── kb_v3.jsonl             ← 451K re-extracted chunks — src/build_kb_v3.py
    │       ├── stix_pairs.jsonl        ← STIX-verified pairs — src/build_stix_pairs.py
    │       ├── se_pairs.jsonl          ← StackExchange Q&A — src/build_se_pairs.py
    │       ├── sft_v3.jsonl            ← final 23,746 pairs — src/build_sft_v3.py
    │       └── dpo_v3.jsonl            ← 3,797 preference pairs — src/build_dpo_data.py
    └── stage*/output/**/adapter_model.safetensors  ← Qwen LoRA adapters (~57 MB each)
```

Public sources (NSL-KDD, UCI SMS, Malimg, MITRE STIX, StackExchange dump,
HackTricks/OWASP clones) can be re-downloaded on any machine. The proprietary
CADRE KB and DFIR-Nexus exports cannot — obtain your own copies.

## Current-line reproduction (SecGPT-Prod on v3 data)

```bash
cd SecGPT-Prod

# 1. Build the v3 dataset (needs ../SecGPTv2/data sources + G: re-extraction inputs)
python src/build_kb_v3.py          # 451K clean chunks from the raw corpus
python src/build_stix_pairs.py     # STIX-verified MITRE pairs
python src/build_se_pairs.py       # StackExchange Q&A (needs Posts.xml dump)
python src/build_sft_v3.py         # assemble data/v3/sft_v3.jsonl (23,746 pairs)

# 2. Train (QLoRA, ~2 h on RTX 4060)
python src/qlora_sft.py --data data/v3/sft_v3.jsonl --output-dir stage1_sft/output/qwen_qlora_v3 --steps 500

# 3. DPO (build pairs, then train ~2.5 h)
python src/build_dpo_data.py --data data/v3/sft_v3.jsonl --out data/v3/dpo_v3.jsonl
python src/dpo_train.py --data data/v3/dpo_v3.jsonl --sft-lora stage1_sft/output/qwen_qlora_v3/checkpoint-500 --output-dir stage2_alignment/output/qwen_dpo_v3

# 4. Benchmark (291 prompts, ~40 min) + stage comparison
python src/eval.py --lora stage2_alignment/output/qwen_dpo_v3/final --name dpov3
python src/eval.py --compare eval/results/<a>.json eval/results/<b>.json
```

## Other pipelines

| Model | Documentation |
|---|---|
| SecGPTv2 (from scratch, 17.4M) | Full 8-step pretrain + SFT + DPO walkthrough in [SecGPTv2/llm_build.md](SecGPTv2/llm_build.md) |
| SecGPTv2.5 (from scratch, 98M) | Corpus → tokenizer → pretrain → SFT → DPO in [SecGPTv2.5/doc.md](SecGPTv2.5/doc.md) |
| SecGPTv3 (GPT-2 fairness run) | Reproduction commands in [SecGPTv3/doc.md](SecGPTv3/doc.md) |
| SecGPT-Prod (Qwen2.5-3B) | Inference, prompts, and retraining in [SecGPT-Prod/USAGE.md](SecGPT-Prod/USAGE.md) |
