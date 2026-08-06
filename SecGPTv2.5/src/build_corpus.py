"""
build_corpus.py — SecGPTv2.5 Step 0: pretraining corpus (~450 MB).

Sources (all v3-era, replacing v2's 77.8 MB corpus built from the incomplete
extraction where only 22% of CADRE chunks were eligible):
  kb    <- SecGPT-Prod/data/v3/kb_v3.jsonl (section-based G: re-extraction,
           stratified per collection to avoid source dominance)
  rule  <- DFIR-Nexus Sigma/Elastic/Splunk/Hayabusa/Chainsaw
  ttp   <- DFIR-Nexus MITRE/CAPEC/CISA/MBC/D3FEND
  ref   <- DFIR-Nexus LOLBAS/GTFOBins/forensics/KAPE/Velociraptor/etc.
  net   <- NSL-KDD lines, spam/ham <- UCI SMS

Output: stage1_pre-training/step0_corpus/output/corpus.txt
Format: one document per block: "<|tag|>\\n<text>\\n\\n"

Usage:
  python src/build_corpus.py [--target-mb 450]
"""

import argparse
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
KB_V3 = ROOT.parent / "SecGPT-Prod" / "data" / "v3" / "kb_v3.jsonl"
DFIR_DIR = ROOT.parent / "SecGPTv2" / "data" / "dfir_nexus_sources"
KDD_FILE = ROOT.parent / "SecGPTv2" / "data" / "KDD+.txt"
SMS_ZIP = ROOT.parent / "SecGPTv2" / "data" / "sms+spam+collection.zip"
OUT = ROOT / "stage1_pre-training" / "step0_corpus" / "output" / "corpus.txt"

SEED = 42
KB_PER_COLLECTION_CAP = 30000

RULE_SOURCES = ["sigma.jsonl", "elastic.jsonl", "splunk_security.jsonl",
                "hayabusa.jsonl", "chainsaw.jsonl"]
TTP_SOURCES = ["mitre_attack.jsonl", "capec.jsonl", "cisa_kev.jsonl",
               "mbc.jsonl", "mitre_d3fend.jsonl"]
REF_SOURCES = ["lolbas.jsonl", "gtfobins.jsonl", "forensic_artifacts.jsonl",
               "kape.jsonl", "velociraptor.jsonl", "hijacklibs.jsonl",
               "loldrivers.jsonl"]


def load_jsonl_texts(path):
    texts = []
    if not path.exists():
        return texts
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                t = json.loads(line).get("text", "")
                if len(t) >= 80:
                    texts.append(t)
            except json.JSONDecodeError:
                continue
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-mb", type=int, default=450)
    args = parser.parse_args()

    rng = random.Random(SEED)
    print("=" * 64)
    print("SecGPTv2.5 — Step 0: Pretraining corpus")
    print("=" * 64)

    docs = []

    print("\n  [kb] kb_v3.jsonl (stratified) ...")
    per_coll = Counter()
    kb_pool = []
    with open(KB_V3, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                kb_pool.append(json.loads(line))
    rng.shuffle(kb_pool)
    for c in kb_pool:
        if per_coll[c["collection"]] >= KB_PER_COLLECTION_CAP:
            continue
        text = c["text"].strip()
        if len(text) >= 200:
            docs.append(("kb", f"{c['title']}\n\n{text}" if c.get("title") else text))
            per_coll[c["collection"]] += 1
    print(f"      {len(docs):,} kb docs across {len(per_coll)} collections")

    for tag, sources in [("rule", RULE_SOURCES), ("ttp", TTP_SOURCES), ("ref", REF_SOURCES)]:
        n0 = len(docs)
        for src in sources:
            for t in load_jsonl_texts(DFIR_DIR / src):
                docs.append((tag, t))
        print(f"  [{tag}] +{len(docs) - n0:,} docs")

    print("  [net] NSL-KDD ...")
    n0 = len(docs)
    if KDD_FILE.exists():
        lines = KDD_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
        rng.shuffle(lines)
        for line in lines[:40000]:
            if line.strip():
                docs.append(("net", line.strip()))
    print(f"      +{len(docs) - n0:,} docs")

    print("  [spam/ham] UCI SMS ...")
    n0 = len(docs)
    if SMS_ZIP.exists():
        import zipfile
        with zipfile.ZipFile(SMS_ZIP) as z:
            with z.open("SMSSpamCollection") as f:
                for raw in f.read().decode("latin-1").splitlines():
                    parts = raw.split("\t", 1)
                    if len(parts) == 2 and parts[0] in ("spam", "ham") and len(parts[1]) >= 10:
                        docs.append((parts[0], parts[1].strip()))
    print(f"      +{len(docs) - n0:,} docs")

    rng.shuffle(docs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    target_bytes = args.target_mb * 1024 * 1024
    written = 0
    kept = Counter()
    with open(OUT, "w", encoding="utf-8") as f:
        for tag, text in docs:
            if written >= target_bytes:
                break
            block = f"<|{tag}|>\n{text}\n\n"
            f.write(block)
            written += len(block.encode("utf-8"))
            kept[tag] += 1

    print(f"\n{'=' * 64}")
    print(f"  Corpus: {OUT}")
    print(f"  Size: {OUT.stat().st_size / 1e6:.0f} MB")
    for tag, c in sorted(kept.items()):
        print(f"    {tag:8s} {c:,} docs")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
