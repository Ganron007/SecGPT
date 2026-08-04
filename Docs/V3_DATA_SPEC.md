# SFT v3 — Data Builder Spec

Status: **spec** (not yet implemented). Written after the base-model control
experiment proved: *template SFT teaches behavior, corrupts facts*
(TTP hallucination 20% base → 87.5% after naive SFT).

## Goals (from benchmark evidence)

| # | Flaw in v1/v2 data | v3 fix |
|---|---|---|
| 1 | Truncated answers teach wrong ID↔content mapping | Complete, sentence-bounded answers (v2 did this — keep) |
| 2 | No verified technique linkage | MITRE STIX relationships + validated IDs only |
| 3 | Template-authored questions (ceiling) | Real Q&A from Security StackExchange |
| 4 | Proprietary CADRE KB (not redistributable) | Open replacements: HackTricks, OWASP, NIST, public reports |
| 5 | 78% of CADRE chunks never eligible (too long) | Re-chunk from raw source `G:\knowledgebase\doc_extract` by section |
| 6 | No hard negatives / verification pairs | Keep + expand v2's verify_yes/verify_no using STIX for ground truth |
| 7 | Near-duplicate rules | Keep v2 dedup, extend to near-dup (fuzzy title match) |

## Source 1: Re-extract CADRE KB from raw (`G:\knowledgebase\doc_extract`)

Current pipeline loss: 483,800 chunks in `cadre_kb.jsonl`, but median chunk is
2,215 chars and our builders filter ≤1,200-1,500 → **only 22.2% (107K) ever
eligible**, and we use 8-12K of those. Raw source is 67 collections
(.md 1.5 GB / .html 38 GB / .txt 1.5 GB, all triples).

New extraction (`src/build_kb_v3.py`):

1. **Section-based chunking**: split `.md` on `##`/`###` headers, target
   800-2,000 chars per chunk (merge small sections, split large at paragraph
   breaks). Every chunk = one topic.
2. **Per-section titles**: nearest header above the chunk (not the course name).
   Fallback: `"{course} — {first-words}"`.
3. **Junk filters**: TOC/nav pages (`## Contents`, `](#page-` density),
   index/glossary sections, pages <100 chars, image/asset-only pages.
4. **Technique validation**: existing `technique_ids` field has garbage
   (e.g. `T0000`, `T1111`). Validate every ID against the real ATT&CK STIX ID
   set; keep only valid ones → enables verified "which technique…" pairs.
5. **Domain-stratified sampling**: use `domain:*` tags to balance
   offensive/defensive/DFIR/cloud/AD across the kb quota.
6. **Cross-collection dedup**: same concept appears in SANS + HTB + CISSP;
   fuzzy-title + first-200-char hash dedup.

Expected yield: ~60-80K clean section chunks (vs 107K noisy page chunks),
of which we sample 10-12K for training. **Licensing unchanged** — still
proprietary, local-only; the open sources below provide the publishable path.

## Source 2: MITRE ATT&CK full STIX (upgrade from flattened jsonl)

Current: `mitre_attack.jsonl` = title + description only.
Upgrade: official STIX bundle (mitre-attack GitHub / TAXII).

New verified pair types (all answers checkable — anti-hallucination):
- technique → **data sources** ("What data sources detect T1059?")
- technique → **mitigations** ("How do you mitigate T1059?")
- technique → **tactics / platforms / sub-techniques**
- tactic → techniques ("Which techniques belong to Defense Evasion?")
- group/software → techniques used
- keep v2's anchored description + by-ID + verify_yes/verify_no pairs,
  now with STIX-verified IDs (no reliance on regex'd text)

Quota: ~6K pairs from STIX + 1.5K verification pairs.

## Source 3: Security StackExchange (real instruction data)

CC-licensed dump (stackexchange / archive.org). Parse `security.stackexchange.com`
Posts.xml:

- accepted answers only, score ≥ 3, question score ≥ 2
- security-relevant tags filter; strip HTML; Q+top answer ≤ 2,000 chars
- dedup, drop image/link-only answers
- Expected: ~10-20K natural Q&A pairs — the only non-synthetic
  instruction data in the corpus. Category: `kb` (kind=`real_qa`).

## Source 4: Open replacements for CADRE KB (publishable corpus path)

| Source | Type | Volume est. | Note |
|---|---|---|---|
| HackTricks (gitbook markdown) | offensive KB | ~5-8K chunks | git repo, free |
| OWASP WSTG + Top 10 + API/MSTG | web/appsec | ~2-3K | CC |
| NIST SP 800-61 / 800-86 / 800-94 + CISA playbooks | IR process | ~1-2K | public domain |
| TheDFIRReport + Unit42 public reports | incident narratives | ~1-2K | scrape, attribution |

These form **`sft_v3_open.jsonl` — fully redistributable** (fixes DATA.md
reproducibility gap; CADRE-derived pairs stay in a separate local-only file).

## Sources kept as-is (already good)

- Rules: Sigma/Elastic/Splunk/Hayabusa/Chainsaw dedup'd (v2) + **full YAML
  with `attack.tXXXX` tags** → verified rule↔technique pairs
- LOLBAS/GTFOBins/forensics/KAPE/Velociraptor/HijackLibs/LOLDrivers
- CISA KEV + CAPEC + MBC + D3FEND; add NVD CVE JSON (bulk, free) ~1-2K vuln pairs
- Classification: UCI SMS + NSL-KDD (unchanged; consider CIC-IDS2017 later)

## Output design

```
data/
├── sft_v3.jsonl        ← full training set (~30K, local-only; includes CADRE)
├── sft_v3_open.jsonl   ← publishable subset (~15-18K, no CADRE/proprietary)
```

Target composition (~30K):
| Category | Pairs | Sources |
|---|---|---|
| ttp | 7.5K | STIX 6K + verify 1.5K |
| rule | 6K | dedup'd rules (v2 method) |
| ref | 3K | LOLBAS etc. |
| kb | 10K | CADRE-v3 6K + StackExchange 3K + open 1K |
| classification | 3.5K | SMS 2K + KDD 1.5K |

## Quality gates (builder asserts)

1. Every MITRE ID in any pair ∈ official STIX ID set
2. No answer >1,600 chars or cut mid-sentence
3. Exact + fuzzy dedup (title + first-200-chars hash)
4. No TOC/nav/boilerplate chunks (regex battery)
5. Per-category + per-kind counts printed; sample 10 random pairs dumped
   for eyeball review before training

## Experiment protocol (unchanged variable isolation)

Train `qwen_qlora_v3` with identical hyperparams (500 steps, lr 2e-4, r16/α32),
benchmark, then 5-way compare:
`base → sft500 (v1 data) → sftv2 → sftv3 → +DPO`.

## Build order & effort

| Step | Effort |
|---|---|
| 1. `build_kb_v3.py` (G: re-chunk + STIX ID validation) | ~3 h scripting |
| 2. STIX fetch + relationship pair generator | ~2 h |
| 3. StackExchange dump parse | ~2 h |
| 4. Open-source scrapers (HackTricks/OWASP/NIST) | ~2-3 h |
| 5. Assemble + quality gates + stats | ~1 h |
| 6. Train (500 steps) + benchmark + compare | ~3 h GPU |
