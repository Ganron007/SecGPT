"""
build_stix_pairs.py — SecGPT-Prod v3: verified MITRE ATT&CK pairs from STIX.

Downloads the official enterprise-attack STIX bundle and builds
relationship-verified pairs (every answer checkable against the graph):
  anchored / by-id descriptions, technique->data sources, technique->
  mitigations, tactic membership, sub-techniques, verify yes/no.

Output: data/v3/stix_pairs.jsonl

Usage:
  python src/build_stix_pairs.py
"""

import io
import json
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "v3"
STIX_CACHE = OUT_DIR / "enterprise-attack.json"
OUT = OUT_DIR / "stix_pairs.jsonl"
STIX_URL = ("https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
            "master/enterprise-attack/enterprise-attack.json")

SEED = 42
MAX_BODY = 1600


def download_stix():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if STIX_CACHE.exists() and STIX_CACHE.stat().st_size > 10_000_000:
        print(f"  Using cached {STIX_CACHE.name} ({STIX_CACHE.stat().st_size / 1e6:.0f} MB)")
        return
    print(f"  Downloading STIX bundle (~80 MB)...")
    urllib.request.urlretrieve(STIX_URL, STIX_CACHE)
    print(f"  Saved: {STIX_CACHE}")


def clean_desc(text):
    text = re.sub(r"\[\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    if len(text) > MAX_BODY:
        cut = text.rfind(". ", 0, MAX_BODY)
        if cut > MAX_BODY * 0.5:
            text = text[: cut + 1]
    return text


def ext_id(obj):
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack" and ref.get("external_id"):
            return ref["external_id"]
    return None


def short(text, limit=200):
    cut = text.rfind(". ", 0, limit)
    if cut < limit * 0.4:
        cut = text.rfind(" ", 0, limit)
    return text[: cut + 1].strip().rstrip(".") + "."


def main():
    rng = random.Random(SEED)
    print("=" * 64)
    print("SecGPT-Prod v3 — MITRE ATT&CK STIX pairs")
    print("=" * 64)

    download_stix()
    bundle = json.loads(STIX_CACHE.read_text(encoding="utf-8"))
    objects = bundle["objects"]
    print(f"  STIX objects: {len(objects):,}")

    techniques = {}
    mitigations = {}
    groups = {}
    software = {}
    relationships = []
    for o in objects:
        if o.get("revoked") or o.get("x_mitre_deprecated"):
            continue
        t = o.get("type")
        if t == "attack-pattern":
            tid = ext_id(o)
            if tid:
                techniques[o["id"]] = {
                    "tid": tid,
                    "name": o.get("name", ""),
                    "desc": clean_desc(o.get("description", "")),
                    "tactics": [p["phase_name"].replace("_", " ").title()
                                for p in o.get("kill_chain_phases", [])],
                    "data_sources": [ds.split(":")[-1].strip()
                                     for ds in o.get("x_mitre_data_sources", [])],
                    "is_sub": o.get("x_mitre_is_subtechnique", False),
                }
        elif t == "course-of-action":
            mitigations[o["id"]] = {"name": o.get("name", ""),
                                    "desc": clean_desc(o.get("description", ""))[:400]}
        elif t == "intrusion-set":
            groups[o["id"]] = o.get("name", "")
        elif t in ("malware", "tool"):
            software[o["id"]] = o.get("name", "")
        elif t == "relationship":
            relationships.append(o)

    print(f"  techniques {len(techniques)} | mitigations {len(mitigations)} | "
          f"groups {len(groups)} | software {len(software)} | relationships {len(relationships):,}")

    tech_mitigations = defaultdict(list)
    group_uses = defaultdict(list)
    sub_of = defaultdict(list)
    for r in relationships:
        src, dst, rt = r.get("source_ref"), r.get("target_ref"), r.get("relationship_type")
        if rt == "mitigates" and dst in techniques and src in mitigations:
            tech_mitigations[dst].append(mitigations[src]["name"])
        elif rt == "uses" and src in groups and dst in techniques:
            group_uses[src].append(techniques[dst])
        elif rt == "uses" and src in software and dst in techniques:
            group_uses[src].append(techniques[dst])
        elif rt == "subtechnique-of" and dst in techniques:
            sub_of[dst].append(techniques.get(src))

    pairs = []

    def add(instruction, response, kind):
        pairs.append({"instruction": instruction, "response": response,
                      "category": "ttp", "kind": kind})

    tech_list = [t for t in techniques.values() if t["desc"] and len(t["desc"]) >= 100]
    rng.shuffle(tech_list)

    for t in tech_list:
        add(f"What is {t['name']} and how do adversaries use it?",
            f"ID: {t['tid']}\n\nDescription: {t['desc']}", "anchored")
    for t in tech_list:
        add(f"Explain MITRE ATT&CK technique {t['tid']}.",
            f"ID: {t['tid']}\n\nDescription: {t['desc']}", "by_id")

    n = 0
    for t in tech_list:
        if t["data_sources"] and n < 1500:
            ds = "\n".join(f"- {d}" for d in t["data_sources"][:8])
            add(f"What data sources can be used to detect {t['name']} ({t['tid']})?",
                f"{t['name']} ({t['tid']}) can be detected using these data sources:\n{ds}",
                "data_sources")
            n += 1

    n = 0
    for tech_id, mits in tech_mitigations.items():
        t = techniques[tech_id]
        if mits and t["desc"] and n < 1500:
            ms = "\n".join(f"- {m}" for m in mits[:6])
            add(f"How can defenders mitigate {t['name']} ({t['tid']})?",
                f"Mitigations for {t['name']} ({t['tid']}):\n{ms}", "mitigation")
            n += 1

    n = 0
    for t in tech_list:
        if t["tactics"] and n < 1000:
            add(f"Which tactic does {t['name']} ({t['tid']}) belong to?",
                f"{t['name']} ({t['tid']}) belongs to: {', '.join(t['tactics'])}.",
                "tactic")
            n += 1

    n = 0
    for parent_id, subs in sub_of.items():
        parent = techniques.get(parent_id)
        subs = [s for s in subs if s]
        if parent and len(subs) >= 2 and parent["desc"] and n < 500:
            lines = "\n".join(f"- {s['tid']}: {s['name']}" for s in subs[:8])
            add(f"What are the sub-techniques of {parent['name']} ({parent['tid']})?",
                f"{parent['name']} ({parent['tid']}) sub-techniques:\n{lines}",
                "subtechniques")
            n += 1

    n = 0
    for name, techs in group_uses.items():
        if len(techs) >= 3 and n < 500:
            lines = "\n".join(f"- {t['tid']}: {t['name']}" for t in techs[:10])
            add(f"What techniques does {name} use?",
                f"{name} is associated with these techniques:\n{lines}", "group_software")
            n += 1

    n = 0
    while n < 1200:
        a, b = rng.choice(tech_list), rng.choice(tech_list)
        if a["tid"] == b["tid"]:
            continue
        if rng.random() < 0.5:
            add(f"Is {a['tid']} {a['name']}?",
                f"Yes. {a['tid']} is {a['name']}: {short(a['desc'])}",
                "verify_yes")
        else:
            add(f"Is {b['tid']} {a['name']}?",
                f"No. {a['name']} is {a['tid']}: {short(a['desc'])} "
                f"{b['tid']} is {b['name']} - a different technique.",
                "verify_no")
        n += 1

    rng.shuffle(pairs)
    with open(OUT, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    kinds = Counter(p["kind"] for p in pairs)
    print(f"\n{'=' * 64}")
    print(f"  STIX pairs: {len(pairs):,} -> {OUT}")
    for k, c in sorted(kinds.items()):
        print(f"    {k:15s} {c:,}")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
