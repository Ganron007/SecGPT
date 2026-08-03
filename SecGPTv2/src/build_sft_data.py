"""
build_sft_data.py — Phase A: Generate SFT (instruction→response) pairs for SecGPT v2.

Creates ~500 Q&A pairs from DFIR-Nexus structured sources + cadre_kb excerpts.
Format: <|tag|>\nQ: {instruction}\nA: {response}\n\n

Usage:
  python src/build_sft_data.py
"""

import io
import json
import random
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DFIR_DIR = DATA / "dfir_nexus_sources"
CADRE_KB = DATA / "cadre_kb.jsonl"
OUTPUT = ROOT / "stage2_sft" / "output"

SEED = 42
MAX_PAIRS = 600
MAX_RESPONSE_LEN = 500


def build_mitre_pairs():
    pairs = []
    f = DFIR_DIR / "mitre_attack.jsonl"
    if not f.exists():
        return pairs
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        text = obj.get("text", "")
        if "MITRE ATT&CK" not in text or len(text) < 80:
            continue
        lines = text.split("\n")
        title = lines[0] if lines else ""
        body = "\n".join(lines[1:]).strip()[:MAX_RESPONSE_LEN]
        if "Technique:" in title:
            q = f"What does {title.replace('MITRE ATT&CK Technique:', '').strip()} do?"
        elif "Mitigation:" in title:
            q = f"How do you mitigate {title.replace('MITRE ATT&CK Mitigation:', '').strip()}?"
        elif "malware" in title.lower():
            q = f"Tell me about {title.replace('MITRE ATT&CK malware', '').strip()}"
        else:
            q = f"Explain: {title.replace('MITRE ATT&CK', '').strip()}"
        if len(body) > 50:
            pairs.append({"tag": "ttp", "instruction": q.strip(), "response": body})
    return pairs


def build_sigma_pairs():
    pairs = []
    f = DFIR_DIR / "sigma.jsonl"
    if not f.exists():
        return pairs
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        text = obj.get("text", "")
        if "Sigma Detection Rule:" not in text or len(text) < 80:
            continue
        lines = text.split("\n")
        title = lines[0].replace("Sigma Detection Rule:", "").strip()
        body = "\n".join(lines[1:]).strip()[:MAX_RESPONSE_LEN]
        q = f"Write a detection rule for: {title}"
        if len(body) > 50:
            pairs.append({"tag": "rule", "instruction": q, "response": body})
    return pairs


def build_lolbas_pairs():
    pairs = []
    f = DFIR_DIR / "lolbas.jsonl"
    if not f.exists():
        return pairs
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        text = obj.get("text", "")
        if "LOLBAS:" not in text or len(text) < 50:
            continue
        lines = text.split("\n")
        title = lines[0].replace("LOLBAS:", "").strip()
        body = "\n".join(lines[1:]).strip()[:MAX_RESPONSE_LEN]
        q = f"How can {title} be abused for living-off-the-land attacks?"
        if len(body) > 30:
            pairs.append({"tag": "ref", "instruction": q, "response": body})
    return pairs


def build_gtfobins_pairs():
    pairs = []
    f = DFIR_DIR / "gtfobins.jsonl"
    if not f.exists():
        return pairs
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        text = obj.get("text", "")
        if "GTFOBins:" not in text or len(text) < 50:
            continue
        lines = text.split("\n")
        title = lines[0].replace("GTFOBins:", "").strip()
        body = "\n".join(lines[1:]).strip()[:MAX_RESPONSE_LEN]
        q = f"How can {title} be used to bypass security restrictions?"
        if len(body) > 30:
            pairs.append({"tag": "ref", "instruction": q, "response": body})
    return pairs


def build_forensic_pairs():
    pairs = []
    f = DFIR_DIR / "forensic_artifacts.jsonl"
    if not f.exists():
        return pairs
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        text = obj.get("text", "")
        if "Forensic Artifact:" not in text or len(text) < 50:
            continue
        lines = text.split("\n")
        title = lines[0].replace("Forensic Artifact:", "").strip()
        body = "\n".join(lines[1:]).strip()[:MAX_RESPONSE_LEN]
        q = f"What is the {title} forensic artifact and why is it useful?"
        if len(body) > 30:
            pairs.append({"tag": "ref", "instruction": q, "response": body})
    return pairs


def build_cisa_pairs():
    pairs = []
    f = DFIR_DIR / "cisa_kev.jsonl"
    if not f.exists():
        return pairs
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        text = obj.get("text", "")
        if "CISA Known Exploited Vulnerability:" not in text or len(text) < 50:
            continue
        lines = text.split("\n")
        title = lines[0].replace("CISA Known Exploited Vulnerability:", "").strip()
        body = "\n".join(lines[1:]).strip()[:MAX_RESPONSE_LEN]
        q = f"What is the vulnerability {title} and how is it exploited?"
        if len(body) > 30:
            pairs.append({"tag": "ttp", "instruction": q, "response": body})
    return pairs


def build_kb_pairs(rng, max_pairs=100):
    pairs = []
    if not CADRE_KB.exists():
        return pairs
    collections_wanted = {"HTB Academy", "SANS", "Malpedia", "Offensive Security", "APTnotes"}
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
            text = obj.get("text", "").strip()
            title = obj.get("title", "")
            if len(text) < 100 or len(text) > 800:
                continue
            if text.startswith("---"):
                continue
            candidates.append({"title": title, "text": text, "collection": obj["collection"]})

    rng.shuffle(candidates)
    templates = [
        "Explain the following security concept: {title}",
        "What does this describe: {title}?",
        "Summarize: {title}",
        "Give an overview of {title}",
    ]
    for c in candidates[:max_pairs]:
        tmpl = rng.choice(templates)
        q = tmpl.format(title=c["title"][:80])
        resp = c["text"][:MAX_RESPONSE_LEN]
        pairs.append({"tag": "kb", "instruction": q, "response": resp})
    return pairs


def main():
    rng = random.Random(SEED)
    print("=" * 60)
    print("Phase A — Building SFT Dataset")
    print("=" * 60)

    all_pairs = []
    print("\n  Generating pairs from structured sources...")
    builders = [
        ("MITRE ATT&CK", build_mitre_pairs),
        ("Sigma Rules", build_sigma_pairs),
        ("LOLBAS", build_lolbas_pairs),
        ("GTFOBins", build_gtfobins_pairs),
        ("Forensic Artifacts", build_forensic_pairs),
        ("CISA KEV", build_cisa_pairs),
    ]
    for name, fn in builders:
        p = fn()
        all_pairs.extend(p)
        print(f"    {name:20s} → {len(p)} pairs")

    print("  Generating KB pairs from cadre_kb...")
    kb = build_kb_pairs(rng, max_pairs=150)
    all_pairs.extend(kb)
    print(f"    {'cadre_kb':20s} → {len(kb)} pairs")

    rng.shuffle(all_pairs)
    all_pairs = all_pairs[:MAX_PAIRS]

    print(f"\n  Total SFT pairs: {len(all_pairs)}")
    from collections import Counter
    tag_dist = Counter(p["tag"] for p in all_pairs)
    for tag, count in tag_dist.most_common():
        print(f"    {tag:10s} {count}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / "sft_data.jsonl"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n  Written: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

    corpus_path = OUTPUT / "sft_corpus.txt"
    with open(corpus_path, "w", encoding="utf-8", newline="\n") as f:
        for p in all_pairs:
            f.write(f"<|{p['tag']}|>\n")
            f.write(f"Q: {p['instruction']}\n")
            f.write(f"A: {p['response']}\n\n")
    print(f"  Written: {corpus_path} ({corpus_path.stat().st_size / 1024:.0f} KB)")

    print(f"\n{'=' * 60}")
    print(f"  DONE — {len(all_pairs)} SFT pairs ready")
    print(f"  Format: <|tag|>\\nQ: {{instruction}}\\nA: {{response}}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
