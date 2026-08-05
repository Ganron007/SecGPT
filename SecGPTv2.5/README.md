# SecGPTv2.5 — 100M From-Scratch Model

The maximum honest from-scratch effort on an RTX 4060 (8 GB VRAM), and the
third model in the final benchmark matrix (vs SecGPTv3/GPT-2 and SecGPT-Prod/Qwen-3B).

- **Architecture:** v2's custom GPT scaled to ~100M (GPT-2-small class: ~12L × 12H × 768d)
- **Pretrain corpus:** 400-500 MB raw security text, built from `SecGPT-Prod/data/v3/kb_v3.jsonl`
  (section-based G: re-extraction) + DFIR-Nexus sources — replacing v2's 77.8 MB corpus
  that was built from the incomplete extraction (22% eligible)
- **Tokenizer:** retrained BPE, 16K vocab (v2's 8K under-compresses)
- **Pipeline:** pretrain → SFT (on `sft_v3.jsonl`) → DPO — same data and recipes as every model
- **Benchmark:** same 291-prompt harness (via custom-format adapter)

Status: ⬜ not started. Build doc will be written live during the build
(what/why/how/alternatives/tradeoffs per the project documentation standard).
