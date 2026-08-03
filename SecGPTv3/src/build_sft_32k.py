"""
build_sft_32k.py — SecGPTv3: Generate ~32K Q&A pairs from full corpus.

Extracts from:
  - DFIR-Nexus structured sources (15K pairs)
  - cadre_kb chunks (12K pairs)
  - SMS classification (2K pairs)
  - NSL-KDD classification (3K pairs)

Usage:
  python src/build_sft_32k.py
  python src/build_sft_32k.py --dry-run
"""

import io
import json
import random
import re
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parent / "SecGPTv2" / "data"
DFIR_DIR = DATA / "dfir_nexus_sources"
CADRE_KB = DATA / "cadre_kb.jsonl"
OUTPUT = ROOT / "data"

SEED = 42
MAX_RESPONSE_LEN = 600
FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def strip_frontmatter(text):
    return FRONTMATTER_RE.sub("", text, count=1).strip()


def load_jsonl(path):
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def gen_mitre():
    pairs = []
    for obj in load_jsonl(DFIR_DIR / "mitre_attack.jsonl"):
        text = obj.get("text", "")
        if len(text) < 80:
            continue
        lines = text.split("\n")
        title = lines[0].replace("MITRE ATT&CK", "").strip().lstrip(":").strip()
        body = "\n".join(lines[1:]).strip()[:MAX_RESPONSE_LEN]
        if len(body) < 50:
            continue
        if "Technique:" in text or "ID: T" in text:
            q = f"What is {title} and how do adversaries use it?"
        elif "Mitigation:" in text:
            q = f"How can defenders mitigate {title}?"
        elif "malware" in text.lower()[:50]:
            q = f"Describe the malware: {title}"
        elif "Group:" in text or "group" in text[:30].lower():
            q = f"Who is {title} and what techniques do they use?"
        else:
            q = f"Explain: {title}"
        pairs.append({"instruction": q, "response": body, "category": "ttp"})
    return pairs


def gen_sigma():
    pairs = []
    for obj in load_jsonl(DFIR_DIR / "sigma.jsonl"):
        text = obj.get("text", "")
        if "Sigma Detection Rule:" not in text or len(text) < 80:
            continue
        lines = text.split("\n")
        title = lines[0].replace("Sigma Detection Rule:", "").strip()
        body = "\n".join(lines[1:]).strip()[:MAX_RESPONSE_LEN]
        if len(body) < 50:
            continue
        templates = [
            f"Write a Sigma detection rule for: {title}",
            f"How would you detect {title}?",
            f"Create a detection rule for the following activity: {title}",
        ]
        pairs.append({"instruction": random.choice(templates), "response": body, "category": "rule"})
    return pairs


def gen_elastic():
    pairs = []
    for obj in load_jsonl(DFIR_DIR / "elastic.jsonl"):
        text = obj.get("text", "")
        if "Elastic Detection Rule:" not in text or len(text) < 80:
            continue
        lines = text.split("\n")
        title = lines[0].replace("Elastic Detection Rule:", "").strip()
        body = "\n".join(lines[1:]).strip()[:MAX_RESPONSE_LEN]
        if len(body) < 50:
            continue
        pairs.append({"instruction": f"Write an Elastic detection rule for: {title}", "response": body, "category": "rule"})
    return pairs


def gen_atomic():
    pairs = []
    for obj in load_jsonl(DFIR_DIR / "atomic.jsonl"):
        text = obj.get("text", "")
        if "Atomic Red Team Test:" not in text or len(text) < 80:
            continue
        lines = text.split("\n")
        title = lines[0].replace("Atomic Red Team Test:", "").strip()
        body = "\n".join(lines[1:]).strip()[:MAX_RESPONSE_LEN]
        if len(body) < 50:
            continue
        pairs.append({"instruction": f"How would you test for: {title}?", "response": body, "category": "rule"})
    return pairs


def gen_capec():
    pairs = []
    for obj in load_jsonl(DFIR_DIR / "capec.jsonl"):
        text = obj.get("text", "")
        if "CAPEC Attack Pattern:" not in text or len(text) < 80:
            continue
        lines = text.split("\n")
        title = lines[0].replace("CAPEC Attack Pattern:", "").strip()
        body = "\n".join(lines[1:]).strip()[:MAX_RESPONSE_LEN]
        if len(body) < 50:
            continue
        pairs.append({"instruction": f"Explain the attack pattern: {title}", "response": body, "category": "ttp"})
    return pairs


def gen_cisa():
    pairs = []
    for obj in load_jsonl(DFIR_DIR / "cisa_kev.jsonl"):
        text = obj.get("text", "")
        if "CISA Known Exploited Vulnerability:" not in text or len(text) < 50:
            continue
        lines = text.split("\n")
        title = lines[0].replace("CISA Known Exploited Vulnerability:", "").strip()
        body = "\n".join(lines[1:]).strip()[:MAX_RESPONSE_LEN]
        if len(body) < 30:
            continue
        pairs.append({"instruction": f"What is {title} and how is it exploited?", "response": body, "category": "ttp"})
    return pairs


def gen_splunk():
    pairs = []
    for obj in load_jsonl(DFIR_DIR / "splunk_security.jsonl"):
        text = obj.get("text", "")
        if "Splunk Detection:" not in text or len(text) < 80:
            continue
        lines = text.split("\n")
        title = lines[0].replace("Splunk Detection:", "").strip()
        body = "\n".join(lines[1:]).strip()[:MAX_RESPONSE_LEN]
        if len(body) < 50:
            continue
        pairs.append({"instruction": f"Write a Splunk detection for: {title}", "response": body, "category": "rule"})
    return pairs


def gen_lolbas():
    pairs = []
    for obj in load_jsonl(DFIR_DIR / "lolbas.jsonl"):
        text = obj.get("text", "")
        if "LOLBAS:" not in text or len(text) < 50:
            continue
        lines = text.split("\n")
        title = lines[0].replace("LOLBAS:", "").strip()
        body = "\n".join(lines[1:]).strip()[:MAX_RESPONSE_LEN]
        if len(body) < 30:
            continue
        pairs.append({"instruction": f"How can {title} be abused in a living-off-the-land attack?", "response": body, "category": "ref"})
    return pairs


def gen_gtfobins():
    pairs = []
    for obj in load_jsonl(DFIR_DIR / "gtfobins.jsonl"):
        text = obj.get("text", "")
        if "GTFOBins:" not in text or len(text) < 50:
            continue
        lines = text.split("\n")
        title = lines[0].replace("GTFOBins:", "").strip()
        body = "\n".join(lines[1:]).strip()[:MAX_RESPONSE_LEN]
        if len(body) < 30:
            continue
        pairs.append({"instruction": f"How can {title} be used to bypass security restrictions on Linux?", "response": body, "category": "ref"})
    return pairs


def gen_forensic():
    pairs = []
    for obj in load_jsonl(DFIR_DIR / "forensic_artifacts.jsonl"):
        text = obj.get("text", "")
        if "Forensic Artifact:" not in text or len(text) < 50:
            continue
        lines = text.split("\n")
        title = lines[0].replace("Forensic Artifact:", "").strip()
        body = "\n".join(lines[1:]).strip()[:MAX_RESPONSE_LEN]
        if len(body) < 30:
            continue
        pairs.append({"instruction": f"What is the {title} forensic artifact and how is it useful in investigations?", "response": body, "category": "ref"})
    return pairs


def gen_kape():
    pairs = []
    for obj in load_jsonl(DFIR_DIR / "kape.jsonl"):
        text = obj.get("text", "")
        if "KAPE Target:" not in text or len(text) < 50:
            continue
        lines = text.split("\n")
        title = lines[0].replace("KAPE Target:", "").strip()
        body = "\n".join(lines[1:]).strip()[:MAX_RESPONSE_LEN]
        if len(body) < 30:
            continue
        pairs.append({"instruction": f"What does the KAPE target {title} collect and why?", "response": body, "category": "ref"})
    return pairs


def gen_velociraptor():
    pairs = []
    for obj in load_jsonl(DFIR_DIR / "velociraptor.jsonl"):
        text = obj.get("text", "")
        if "Velociraptor Artifact:" not in text or len(text) < 50:
            continue
        lines = text.split("\n")
        title = lines[0].replace("Velociraptor Artifact:", "").strip()
        body = "\n".join(lines[1:]).strip()[:MAX_RESPONSE_LEN]
        if len(body) < 30:
            continue
        pairs.append({"instruction": f"What does the Velociraptor artifact {title} do?", "response": body, "category": "ref"})
    return pairs


def gen_hijacklibs():
    pairs = []
    for obj in load_jsonl(DFIR_DIR / "hijacklibs.jsonl"):
        text = obj.get("text", "")
        if "HijackLibs DLL Hijack:" not in text or len(text) < 50:
            continue
        lines = text.split("\n")
        title = lines[0].replace("HijackLibs DLL Hijack:", "").strip()
        body = "\n".join(lines[1:]).strip()[:MAX_RESPONSE_LEN]
        if len(body) < 30:
            continue
        pairs.append({"instruction": f"Explain the DLL hijacking vector: {title}", "response": body, "category": "ref"})
    return pairs


def gen_loldrivers():
    pairs = []
    for obj in load_jsonl(DFIR_DIR / "loldrivers.jsonl"):
        text = obj.get("text", "")
        if "LOLDrivers Vulnerable Driver:" not in text or len(text) < 50:
            continue
        lines = text.split("\n")
        title = lines[0].replace("LOLDrivers Vulnerable Driver:", "").strip()
        body = "\n".join(lines[1:]).strip()[:MAX_RESPONSE_LEN]
        if len(body) < 30:
            continue
        pairs.append({"instruction": f"What is the vulnerable driver {title} and how can it be exploited?", "response": body, "category": "ref"})
    return pairs


def gen_mbc():
    pairs = []
    for obj in load_jsonl(DFIR_DIR / "mbc.jsonl"):
        text = obj.get("text", "")
        if "MBC" not in text or len(text) < 50:
            continue
        body = text.strip()[:MAX_RESPONSE_LEN]
        if "Objective:" in text or "Behavior:" in text:
            lines = text.split("\n")
            title = lines[0].replace("MBC Malware Objective:", "").replace("MBC Malware Behavior:", "").strip()
            pairs.append({"instruction": f"Describe the malware behavior: {title}", "response": body, "category": "ttp"})
    return pairs


def gen_d3fend():
    pairs = []
    for obj in load_jsonl(DFIR_DIR / "mitre_d3fend.jsonl"):
        text = obj.get("text", "")
        if "D3FEND" not in text or len(text) < 50:
            continue
        lines = text.split("\n")
        title = lines[0].replace("D3FEND Defensive Technique:", "").strip()
        body = "\n".join(lines[1:]).strip()[:MAX_RESPONSE_LEN]
        if len(body) < 30:
            continue
        pairs.append({"instruction": f"What is the defensive technique: {title}?", "response": body, "category": "ttp"})
    return pairs


def gen_hayabusa():
    pairs = []
    for obj in load_jsonl(DFIR_DIR / "hayabusa.jsonl"):
        text = obj.get("text", "")
        if "Hayabusa Detection Rule:" not in text or len(text) < 50:
            continue
        lines = text.split("\n")
        title = lines[0].replace("Hayabusa Detection Rule:", "").strip()
        body = "\n".join(lines[1:]).strip()[:MAX_RESPONSE_LEN]
        if len(body) < 30:
            continue
        pairs.append({"instruction": f"Write a Hayabusa detection rule for: {title}", "response": body, "category": "rule"})
    return pairs


def gen_chainsaw():
    pairs = []
    for obj in load_jsonl(DFIR_DIR / "chainsaw.jsonl"):
        text = obj.get("text", "")
        if "Chainsaw" not in text or len(text) < 50:
            continue
        lines = text.split("\n")
        title = lines[0].replace("Chainsaw MFT Detection Rule:", "").strip()
        body = "\n".join(lines[1:]).strip()[:MAX_RESPONSE_LEN]
        if len(body) < 30:
            continue
        pairs.append({"instruction": f"How would you detect {title} using Chainsaw?", "response": body, "category": "rule"})
    return pairs


def gen_kb(rng, max_pairs=12000):
    pairs = []
    if not CADRE_KB.exists():
        return pairs
    collections_wanted = {
        "HTB Academy", "SANS", "Malpedia", "Offensive Security", "APTnotes",
        "eLearnSecurity", "Mandiant Enterprise IR", "Kaspersky", "FLARE Learning Hub",
        "Maldev Academy", "SpecterOps", "Black Hat", "DEF CON 33 2025",
        "Altered Security", "ZeroPoint Security", "CISSP", "AI Engineering",
    }
    candidates = []
    with open(CADRE_KB, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("collection") not in collections_wanted:
                continue
            text = strip_frontmatter(obj.get("text", ""))
            title = obj.get("title", "")
            if len(text) < 100 or len(text) > 1200:
                continue
            if text.startswith("#") and len(text) < 150:
                continue
            candidates.append({"title": title, "text": text, "collection": obj["collection"]})

    rng.shuffle(candidates)
    templates = [
        "Explain the following security concept: {title}",
        "What does this describe: {title}?",
        "Provide an overview of: {title}",
        "Summarize the key points about: {title}",
        "What is {title} and why is it important in cybersecurity?",
        "Describe {title} in detail.",
    ]
    for c in candidates[:max_pairs]:
        tmpl = rng.choice(templates)
        q = tmpl.format(title=c["title"][:100])
        resp = c["text"][:MAX_RESPONSE_LEN]
        pairs.append({"instruction": q, "response": resp, "category": "kb"})
    return pairs


def gen_sms(rng, max_pairs=2000):
    import zipfile
    pairs = []
    sms_zip = DATA / "sms+spam+collection.zip"
    if not sms_zip.exists():
        return pairs
    with zipfile.ZipFile(sms_zip) as z:
        with z.open("SMSSpamCollection") as f:
            lines = f.read().decode("latin-1").splitlines()
    rng.shuffle(lines)
    for raw in lines[:max_pairs]:
        parts = raw.split("\t", 1)
        if len(parts) != 2:
            continue
        label, msg = parts[0].strip(), parts[1].strip()
        if len(msg) < 10:
            continue
        if label == "spam":
            q = "Classify this SMS message as spam or ham. Explain why.\nMessage: " + msg
            r = f"This is SPAM. Indicators: unsolicited offer, urgency, suspicious links/numbers, too-good-to-be-true claims.\n\nMessage: \"{msg}\""
        else:
            q = "Classify this SMS message as spam or ham. Explain why.\nMessage: " + msg
            r = f"This is HAM (legitimate). Indicators: personal tone, conversational language, no suspicious offers or urgency.\n\nMessage: \"{msg}\""
        pairs.append({"instruction": q, "response": r, "category": "classification"})
    return pairs


def gen_kdd(rng, max_pairs=3000):
    pairs = []
    kdd_file = DATA / "KDD+.txt"
    if not kdd_file.exists():
        return pairs
    lines = kdd_file.read_text(encoding="utf-8", errors="replace").splitlines()
    rng.shuffle(lines)
    attack_types = {}
    for line in lines[:max_pairs * 3]:
        parts = line.strip().split(",")
        if len(parts) < 42:
            continue
        label = parts[-2]
        if label not in attack_types:
            attack_types[label] = []
        attack_types[label].append(line.strip())

    count = 0
    for label, records in attack_types.items():
        for rec in records[:max_pairs // len(attack_types)]:
            parts = rec.split(",")
            proto, service, flag = parts[1], parts[2], parts[3]
            if label == "normal":
                q = f"Analyze this network connection and classify it.\nConnection: {rec}"
                r = f"This is NORMAL traffic. Protocol: {proto}, Service: {service}, Flag: {flag}. No attack indicators present."
            else:
                q = f"Analyze this network connection and classify it.\nConnection: {rec}"
                r = f"This is an ATTACK: {label}. Protocol: {proto}, Service: {service}, Flag: {flag}. Attack category: {label}."
            pairs.append({"instruction": q, "response": r, "category": "classification"})
            count += 1
            if count >= max_pairs:
                break
        if count >= max_pairs:
            break
    return pairs


def main():
    rng = random.Random(SEED)
    print("=" * 60)
    print("SecGPTv3 — Generating ~32K SFT Dataset")
    print("=" * 60)

    generators = [
        ("MITRE ATT&CK", gen_mitre),
        ("Sigma Rules", gen_sigma),
        ("Elastic Rules", gen_elastic),
        ("Atomic Red Team", gen_atomic),
        ("CAPEC", gen_capec),
        ("CISA KEV", gen_cisa),
        ("Splunk", gen_splunk),
        ("Hayabusa", gen_hayabusa),
        ("Chainsaw", gen_chainsaw),
        ("LOLBAS", gen_lolbas),
        ("GTFOBins", gen_gtfobins),
        ("Forensic Artifacts", gen_forensic),
        ("KAPE", gen_kape),
        ("Velociraptor", gen_velociraptor),
        ("HijackLibs", gen_hijacklibs),
        ("LOLDrivers", gen_loldrivers),
        ("MBC", gen_mbc),
        ("D3FEND", gen_d3fend),
    ]

    all_pairs = []
    print("\n  DFIR-Nexus structured sources:")
    for name, fn in generators:
        p = fn()
        all_pairs.extend(p)
        print(f"    {name:20s} → {len(p):>5,} pairs")

    print("\n  Knowledge base (cadre_kb):")
    kb = gen_kb(rng)
    all_pairs.extend(kb)
    print(f"    {'cadre_kb':20s} → {len(kb):>5,} pairs")

    print("\n  Classification tasks:")
    sms = gen_sms(rng)
    all_pairs.extend(sms)
    print(f"    {'SMS spam/ham':20s} → {len(sms):>5,} pairs")

    kdd = gen_kdd(rng)
    all_pairs.extend(kdd)
    print(f"    {'NSL-KDD':20s} → {len(kdd):>5,} pairs")

    rng.shuffle(all_pairs)

    from collections import Counter
    cat_dist = Counter(p["category"] for p in all_pairs)
    print(f"\n  TOTAL: {len(all_pairs):,} pairs")
    print(f"\n  By category:")
    for cat, count in cat_dist.most_common():
        print(f"    {cat:15s} {count:>6,} ({count/len(all_pairs)*100:.1f}%)")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / "sft_32k.jsonl"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n  Written: {out_path} ({size_mb:.1f} MB)")
    print(f"\n{'=' * 60}")
    print(f"  DONE — {len(all_pairs):,} SFT pairs ready for Qwen2.5-3B QLoRA")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
