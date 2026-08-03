"""
build_corpus.py — SecGPT v2 Step 0: Larger corpus assembly (~50 MB).

Same pipeline as v1 but 10× larger samples. Targets:
  kb: 30,000 chunks (~30 MB)
  rule: 5,000 records (~5 MB)
  ttp: 3,000 records (~3 MB)
  ref: 2,000 records (~3 MB)
  net: 30,000 lines (~4 MB)
  spam/ham: all available
  malware: 1,000 images

Usage:
  python src/build_corpus.py [--seed 42] [--dry-run]
"""

import argparse
import hashlib
import io
import json
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

CADRE_KB = DATA / "cadre_kb.jsonl"
DFIR_DIR = DATA / "dfir_nexus_sources"
SMS_ZIP = DATA / "sms+spam+collection.zip"
KDD_FILE = DATA / "KDD+.txt"
MALIMG_ZIP = DATA / "malimg.zip"
MALIMG_DIR = DATA / "malimg_paper_dataset_imgs"

CORPUS_OUT = ROOT / "stage1_pre-training" / "step0_corpus" / "output" / "corpus.txt"

CADRE_TAG_MAP = {
    "eLearnSecurity": "kb", "Mandiant Enterprise IR": "kb", "SANS": "kb",
    "EBooks": "kb", "Malpedia": "kb", "AI Engineering": "kb",
    "Conference Materials": "kb", "APTnotes": "kb", "CISSP": "kb",
    "HTB Academy": "kb", "Offensive Security": "kb", "Altered Security": "kb",
    "Reversing UAL Dennis Yurichev": "kb", "DEF CON 33 2025": "kb",
    "HackSys Windows Kernel Exploitation": "kb",
    "HackSys Windows Kernel Exploitation 2023": "kb",
    "SpecterOps": "kb", "Black Hat": "kb", "ZeroPoint Security": "kb",
    "Maldev Academy": "kb", "Forty North": "kb", "CCSK": "kb",
    "DFRWS EU 2026": "kb", "Volatility Malware & Memory Training": "kb",
    "CyberWarfareLabs": "kb", "Zero2Auto Reverse Engineering": "kb",
    "MalTrak": "kb", "Kaspersky": "kb", "CIS Controls v8": "kb",
    "SilentBreak DarkSide Ops Malware Dev": "kb", "Mastering Burp Suite Pro": "kb",
    "Cyber Plumber": "kb", "TrainSec": "kb", "EvilGoPhish Mastery": "kb",
    "Mandiant Malware Analysis": "kb", "MDSec Adversary Simulation": "kb",
    "Red Team Training Mr.Un1k0d3r": "kb", "Evilginx Masterclass": "kb",
    "FLARE Learning Hub": "kb", "Hexorcist": "kb",
    "Hands-On KQL for Security Analysts": "kb", "IntelTechniques OSINT": "kb",
    "HackTricks AWS Red Team Expert": "kb", "Dark Vortex Malware on Steroids": "kb",
    "Linux Forensics": "kb", "Memory Forensics Masterclass": "kb",
    "Dark Vortex Red Team OpSec": "kb", "BC Security Empire Operations": "kb",
    "IntelTechniques Privacy": "kb", "Red Siege Kerberos Workshop": "kb",
    "AD Attacks Reference": "kb", "Malicious Packet Analysis": "kb",
    "CodeMachine Kernel Rootkits": "kb", "SentinelOne - Threat Hunting course": "kb",
    "Maldev1": "kb", "Sektor7": "kb", "Brett Case Studies": "kb",
    "Certified CyberDefender Blue Team": "kb", "SentinelOne Incident Response": "kb",
    "Signal Labs Vulnerability Research": "kb", "XINTRA Azure & M365": "kb",
    "RE504": "kb", "AD Mindmap": "kb",
    "Signal Labs Offensive Tool Development": "kb",
    "DFIR Adversarial Lab Guide": "kb", "Practical TLS": "kb",
    "YARA Rules": "rule", "CAPA Rules": "rule",
    "MITRE ATT&CK": "ttp", "CAPEC": "ttp", "MBC": "ttp",
    "WinLOLBIN-GT": "ref", "Bazaar 2026-06": "ref",
    "Jeff Nippard": None,
}

DFIR_TAG_MAP = {
    "sigma": "rule", "elastic": "rule", "atomic": "rule",
    "chainsaw": "rule", "hayabusa": "rule", "splunk_security": "rule",
    "mitre_attack": "ttp", "mitre_atlas": "ttp", "mitre_car": "ttp",
    "mitre_d3fend": "ttp", "mitre_engage": "ttp", "capec": "ttp",
    "cisa_kev": "ttp", "mbc": "ttp",
    "lolbas": "ref", "gtfobins": "ref", "hijacklibs": "ref",
    "loldrivers": "ref", "forensic_artifacts": "ref",
    "forensic_clarifications": "ref", "kape": "ref", "velociraptor": "ref",
    "stratus_red_team": "kb",
}

SAMPLE_TARGETS = {
    "kb": 30000,
    "rule": 5000,
    "ttp": 3000,
    "ref": 2000,
    "spam": 99999,
    "ham": 99999,
    "net": 30000,
    "malware": 1000,
}

KB_WEIGHTS = {
    "HTB Academy": 5.0, "SANS": 3.0, "Malpedia": 3.0,
    "Mandiant Enterprise IR": 2.0, "eLearnSecurity": 2.0,
    "Offensive Security": 2.0, "APTnotes": 2.0,
    "FLARE Learning Hub": 2.0, "Kaspersky": 2.0,
}

RULE_WEIGHTS = {
    "YARA Rules": 3.0, "sigma": 3.0, "elastic": 2.0,
    "atomic": 2.0, "CAPA Rules": 1.5, "splunk_security": 1.5,
}

MIN_CHARS = 50
MAX_CHARS = 4000
MALWARE_HEX_BYTES = 200
FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def strip_frontmatter(text):
    return FRONTMATTER_RE.sub("", text, count=1).strip()


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def text_hash(text):
    return hashlib.sha256(normalize(text).encode("utf-8", errors="replace")).hexdigest()


def ingest_cadre_kb():
    records = []
    if not CADRE_KB.exists():
        print(f"  [SKIP] {CADRE_KB} not found")
        return records
    with open(CADRE_KB, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            collection = obj.get("collection", "")
            tag = CADRE_TAG_MAP.get(collection)
            if tag is None:
                continue
            text = strip_frontmatter(obj.get("text", ""))
            if len(text) < MIN_CHARS:
                continue
            text = text[:MAX_CHARS]
            records.append({"tag": tag, "text": text, "source": f"cadre_kb/{collection}"})
    return records


def ingest_dfir():
    records = []
    if not DFIR_DIR.exists():
        print(f"  [SKIP] {DFIR_DIR} not found")
        return records
    for jf in sorted(DFIR_DIR.glob("*.jsonl")):
        stem = jf.stem
        tag = DFIR_TAG_MAP.get(stem)
        if tag is None:
            continue
        with open(jf, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text", "").strip()
                if len(text) < MIN_CHARS:
                    continue
                text = text[:MAX_CHARS]
                records.append({"tag": tag, "text": text, "source": f"dfir/{stem}"})
    return records


def ingest_sms():
    records = []
    if not SMS_ZIP.exists():
        print(f"  [SKIP] {SMS_ZIP} not found")
        return records
    with zipfile.ZipFile(SMS_ZIP) as z:
        with z.open("SMSSpamCollection") as f:
            for raw in f.read().decode("latin-1").splitlines():
                parts = raw.split("\t", 1)
                if len(parts) != 2:
                    continue
                label, msg = parts[0].strip(), parts[1].strip()
                if label not in ("ham", "spam"):
                    continue
                if len(msg) < 10:
                    continue
                records.append({"tag": label, "text": msg, "source": "uci_sms"})
    return records


def ingest_kdd():
    records = []
    if not KDD_FILE.exists():
        print(f"  [SKIP] {KDD_FILE} not found")
        return records
    with open(KDD_FILE, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append({"tag": "net", "text": line, "source": "nsl_kdd"})
    return records


def ingest_malimg():
    records = []
    if not MALIMG_DIR.exists() and MALIMG_ZIP.exists():
        print(f"  Extracting {MALIMG_ZIP} ...")
        with zipfile.ZipFile(MALIMG_ZIP) as z:
            z.extractall(DATA)
    if not MALIMG_DIR.exists():
        print(f"  [SKIP] {MALIMG_DIR} not found")
        return records
    families = sorted([d for d in MALIMG_DIR.iterdir() if d.is_dir()])
    per_family = max(1, SAMPLE_TARGETS["malware"] // len(families))
    for fam in families:
        pngs = sorted(fam.glob("*.png"))[:per_family]
        for png in pngs:
            raw = png.read_bytes()[:MALWARE_HEX_BYTES]
            text = f"family:{fam.name} bytes:{raw.hex()}"
            records.append({"tag": "malware", "text": text, "source": f"malimg/{fam.name}"})
    return records


def deduplicate(records):
    seen = set()
    unique = []
    dupes = 0
    for rec in records:
        h = text_hash(rec["text"])
        if h in seen:
            dupes += 1
            continue
        seen.add(h)
        unique.append(rec)
    print(f"  Dedup: {len(records)} -> {len(unique)} unique ({dupes} removed)")
    return unique


def weighted_sample(records, tag, n, rng):
    pool = [r for r in records if r["tag"] == tag]
    if len(pool) <= n:
        return pool
    if tag == "kb" and KB_WEIGHTS:
        weights = [KB_WEIGHTS.get(r["source"].split("/", 1)[-1], 1.0) for r in pool]
    elif tag == "rule" and RULE_WEIGHTS:
        weights = [RULE_WEIGHTS.get(r["source"].split("/", 1)[-1], 1.0) for r in pool]
    else:
        weights = None
    if weights:
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        indices = set()
        attempts = 0
        while len(indices) < n and attempts < n * 10:
            r = rng.random()
            cumulative = 0.0
            for i, p in enumerate(probs):
                cumulative += p
                if r <= cumulative:
                    indices.add(i)
                    break
            attempts += 1
        return [pool[i] for i in sorted(indices)]
    else:
        return rng.sample(pool, n)


def build_corpus(seed=42, dry_run=False):
    rng = random.Random(seed)
    print("=" * 60)
    print("SecGPT v2 — Step 0: Corpus Assembly (~50 MB target)")
    print("=" * 60)

    print("\n[1/6] Ingesting cadre_kb.jsonl...")
    cadre = ingest_cadre_kb()
    print(f"  -> {len(cadre)} records")

    print("\n[2/6] Ingesting DFIR-Nexus sources...")
    dfir = ingest_dfir()
    print(f"  -> {len(dfir)} records")

    print("\n[3/6] Ingesting SMS collection...")
    sms = ingest_sms()
    print(f"  -> {len(sms)} records")

    print("\n[4/6] Ingesting NSL-KDD...")
    kdd = ingest_kdd()
    print(f"  -> {len(kdd)} records")

    print("\n[5/6] Ingesting Malimg hex bytes...")
    malimg = ingest_malimg()
    print(f"  -> {len(malimg)} records")

    all_records = cadre + dfir + sms + kdd + malimg
    print(f"\n  Total ingested: {len(all_records)}")

    print("\n[6/6] Deduplicating...")
    all_records = deduplicate(all_records)

    tag_counts = Counter(r["tag"] for r in all_records)
    print("\n  Records per tag (after dedup):")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"    {tag:10s} {count:>8,}")

    print("\n  Sampling...")
    sampled = []
    for tag, target in SAMPLE_TARGETS.items():
        chunk = weighted_sample(all_records, tag, target, rng)
        sampled.extend(chunk)
        print(f"    {tag:10s} {len(chunk):>6,} / {target:,} target")

    rng.shuffle(sampled)
    total_chars = sum(len(r["text"]) for r in sampled)
    print(f"\n  Sampled: {len(sampled)} records, {total_chars:,} chars ({total_chars / 1024 / 1024:.2f} MB)")

    if dry_run:
        print("\n  [DRY RUN] No files written.")
        return

    CORPUS_OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n  Writing {CORPUS_OUT}...")
    with open(CORPUS_OUT, "w", encoding="utf-8", newline="\n") as f:
        for rec in sampled:
            f.write(f"<|{rec['tag']}|>\n")
            f.write(rec["text"] + "\n\n")

    corpus_size = CORPUS_OUT.stat().st_size
    print(f"\n{'=' * 60}")
    print(f"  DONE")
    print(f"  corpus.txt: {corpus_size:,} bytes ({corpus_size / 1024 / 1024:.2f} MB)")
    print(f"  records:    {len(sampled)}")
    print(f"{'=' * 60}")

    print("\n  Tag distribution:")
    final_chars = defaultdict(int)
    final_counts = Counter(r["tag"] for r in sampled)
    for r in sampled:
        final_chars[r["tag"]] += len(r["text"])
    for tag in sorted(final_counts, key=lambda t: -final_chars[t]):
        pct = final_chars[tag] / total_chars * 100
        print(f"    {tag:10s} {final_counts[tag]:>6,} records  {final_chars[tag]:>10,} chars  ({pct:5.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    build_corpus(seed=args.seed, dry_run=args.dry_run)
