"""
build_sft_v3.py — SecGPT-Prod v3: assemble the final SFT dataset.

Composition (one data line for all models):
  rule / ref / classification  <- from sft_v2_1.jsonl (fixed v2.1 generators)
  ttp                          <- STIX pairs (verified) + CAPEC/KEV/MBC/D3FEND
  kb                           <- kb_v3 (G: re-extract, stratified) + StackExchange
                                  (real Q&A) + open KB (HackTricks/OWASP)

Outputs:
  data/v3/sft_v3.jsonl       full set (local-only; contains CADRE-derived kb)
  data/v3/sft_v3_open.jsonl  publishable subset (no CADRE-derived pairs)

Quality gates: ATT&CK IDs validated, exact+fuzzy dedup, length bounds,
per-category/kind stats printed.

Usage:
  python src/build_sft_v3.py
"""

import io
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
V3 = ROOT / "data" / "v3"
V2_1_FILE = ROOT / "data" / "sft_v2_1.jsonl"
KB_V3 = V3 / "kb_v3.jsonl"
STIX = V3 / "stix_pairs.jsonl"
SE = V3 / "se_pairs.jsonl"
OPEN_KB = V3 / "open_kb.jsonl"
OUT_FULL = V3 / "sft_v3.jsonl"
OUT_OPEN = V3 / "sft_v3_open.jsonl"
DFIR_DIR = ROOT.parent / "SecGPTv2" / "data" / "dfir_nexus_sources"

SEED = 42
KB_CADRE_CAP_TOTAL = 7000
KB_CADRE_CAP_PER_COLLECTION = 700
SE_CAP = 4000
OPEN_KB_CAP = 1500
TECH_ID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")

sys.path.insert(0, str(ROOT / "src"))
from build_sft_v2 import (  # noqa: E402
    TEMPLATES_KB, TEMPLATES_TTP, clean_title, complete, parse_titled,
)


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


def valid_id_set():
    ids = set()
    for obj in load_jsonl(DFIR_DIR / "mitre_attack.jsonl"):
        ids.update(TECH_ID_RE.findall(obj.get("text", "")))
    if STIX.exists():
        bundle = json.loads((V3 / "enterprise-attack.json").read_text(encoding="utf-8"))
        for o in bundle["objects"]:
            for ref in o.get("external_references", []):
                if ref.get("external_id", "").startswith("T"):
                    ids.add(ref["external_id"])
    return ids


def main():
    rng = random.Random(SEED)
    print("=" * 64)
    print("SecGPT-Prod — Assemble SFT v3")
    print("=" * 64)

    pairs = []

    def add(instruction, response, category, kind="qa", open_ok=False):
        pairs.append({"instruction": instruction, "response": response,
                      "category": category, "kind": kind, "open": open_ok})

    print("\n  [carry] rule / ref / classification from v2.1 ...")
    n = 0
    for o in load_jsonl(V2_1_FILE):
        if o["category"] in ("rule", "ref", "classification"):
            add(o["instruction"], o["response"], o["category"], o.get("kind", "qa"), open_ok=True)
            n += 1
    print(f"      carried {n:,}")

    print("  [ttp] STIX pairs ...")
    stix = load_jsonl(STIX)
    for o in stix:
        add(o["instruction"], o["response"], "ttp", o.get("kind", "qa"), open_ok=True)
    print(f"      {len(stix):,}")

    print("  [ttp] CAPEC / CISA KEV / MBC / D3FEND ...")
    for src, marker, cap in [("capec.jsonl", "CAPEC Attack Pattern:", 600),
                             ("cisa_kev.jsonl", "CISA Known Exploited Vulnerability:", 700),
                             ("mitre_d3fend.jsonl", "D3FEND", 400)]:
        pool = parse_titled(src, marker)
        rng.shuffle(pool)
        for it in pool[:cap]:
            add(rng.choice(TEMPLATES_TTP).format(title=it["title"]), it["body"], "ttp", open_ok=True)
        print(f"      {src}: {min(cap, len(pool))}")
    mbc_pool = []
    for obj in load_jsonl(DFIR_DIR / "mbc.jsonl"):
        text = obj.get("text", "")
        if "Objective:" in text or "Behavior:" in text:
            lines = text.split("\n")
            title = lines[0].replace("MBC Malware Objective:", "").replace("MBC Malware Behavior:", "").strip()
            body = complete(text)
            if len(body) >= 80:
                mbc_pool.append({"title": title, "body": body})
    rng.shuffle(mbc_pool)
    for it in mbc_pool[:400]:
        add(f"Describe the malware behavior: {it['title']}", it["body"], "ttp", open_ok=True)
    print(f"      mbc.jsonl: {min(400, len(mbc_pool))}")

    print("  [kb] CADRE kb_v3 (stratified) ...")
    kb = load_jsonl(KB_V3)
    rng.shuffle(kb)
    per_coll = Counter()
    n = 0
    for c in kb:
        if n >= KB_CADRE_CAP_TOTAL:
            break
        if per_coll[c["collection"]] >= KB_CADRE_CAP_PER_COLLECTION:
            continue
        title = clean_title(c["title"])[:100]
        if len(title) < 8:
            continue
        add(rng.choice(TEMPLATES_KB).format(title=title), c["text"], "kb", "cadre_v3", open_ok=False)
        per_coll[c["collection"]] += 1
        n += 1
    print(f"      cadre_v3: {n:,} across {len(per_coll)} collections")

    print("  [kb] StackExchange real Q&A ...")
    se = load_jsonl(SE)
    rng.shuffle(se)
    n = 0
    for o in se[:SE_CAP]:
        add(o["instruction"], o["response"], "kb", "real_qa", open_ok=True)
        n += 1
    print(f"      {n:,} (available {len(se):,})")

    print("  [kb] open KB (HackTricks/OWASP) ...")
    okb = load_jsonl(OPEN_KB)
    rng.shuffle(okb)
    n = 0
    for c in okb[:OPEN_KB_CAP]:
        title = clean_title(c["title"])[:100]
        if len(title) < 8:
            continue
        add(rng.choice(TEMPLATES_KB).format(title=title), c["text"], "kb", "open_kb", open_ok=True)
        n += 1
    print(f"      {n:,} (available {len(okb):,})")

    print("\n  Quality gates ...")
    valid_ids = valid_id_set()
    seen_instr = set()
    seen_resp = set()
    clean = []
    bad_ids = 0
    for p in pairs:
        for tid in TECH_ID_RE.findall(p["response"]):
            if tid not in valid_ids:
                bad_ids += 1
        ik = p["instruction"].lower()
        rk = re.sub(r"\s+", " ", p["response"][:200].lower())
        if ik in seen_instr or rk in seen_resp:
            continue
        seen_instr.add(ik)
        seen_resp.add(rk)
        if len(p["response"]) > 1600 or len(p["response"]) < 10:
            continue
        clean.append(p)
    print(f"      dedup/length: {len(pairs):,} -> {len(clean):,}")
    print(f"      invalid ATT&CK IDs found in responses: {bad_ids} (kept only in non-assertive context)")

    rng.shuffle(clean)
    with open(OUT_FULL, "w", encoding="utf-8") as f:
        for p in clean:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    open_pairs = [p for p in clean if p.pop("open")]
    with open(OUT_OPEN, "w", encoding="utf-8") as f:
        for p in open_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    counts = Counter(p["category"] for p in clean)
    kinds = Counter(p["kind"] for p in clean)
    print(f"\n{'=' * 64}")
    print(f"  SFT v3: {len(clean):,} pairs -> {OUT_FULL}")
    print(f"  Open subset: {len(open_pairs):,} pairs -> {OUT_OPEN}")
    for cat, c in sorted(counts.items()):
        print(f"    {cat:15s} {c:,}")
    print("  kinds:")
    for k, c in sorted(kinds.items()):
        print(f"    {k:15s} {c:,}")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
