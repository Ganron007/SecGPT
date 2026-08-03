"""
build_eval_set.py — SecGPT-Prod: build the benchmark evaluation set.

~250 prompts with ground truth, stratified across categories:
  ttp 50 / rule 50 / ref 50 / kb 40 / classification 50

Leakage policy: any source record present in data/sft_32k.jsonl is flagged
(meta.leaked). DFIR-Nexus sources were fully consumed by training, so their
prompts measure trained-knowledge RECALL; kb/SMS/KDD have held-out records
and measure GENERALIZATION. eval.py reports both splits.

Output: eval/eval_set.jsonl  (gitignored via *.jsonl)

Usage:
  python src/build_eval_set.py
"""

import io
import json
import random
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parent / "SecGPTv2" / "data"
DFIR_DIR = DATA / "dfir_nexus_sources"
CADRE_KB = DATA / "cadre_kb.jsonl"
TRAIN_FILE = ROOT / "data" / "sft_32k.jsonl"
OUT_DIR = ROOT / "eval"
OUT_FILE = OUT_DIR / "eval_set.jsonl"

SEED = 42
FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
TECH_ID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
WORD_RE = re.compile(r"[a-z][a-z0-9_-]{3,}")

STOPWORDS = {
    "this", "that", "with", "from", "they", "them", "their", "have", "has",
    "been", "were", "will", "would", "could", "should", "into", "when",
    "where", "which", "what", "adversaries", "attacker", "attackers",
    "using", "used", "uses", "use", "via", "also", "such", "may", "can",
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "each",
    "other", "than", "then", "these", "those", "about", "after", "before",
    "between", "through", "during", "under", "over", "against", "within",
    "technique", "detection", "rule", "following", "describe", "explain",
    "https", "http", "www", "com", "org", "net", "blog", "twitter",
    "github", "reference", "references", "source", "status", "level",
    "author", "date", "modified", "false", "true", "none", "description",
}


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


def keywords(text, top_n=8):
    words = WORD_RE.findall(text.lower())
    freq = Counter(w for w in words if w not in STOPWORDS)
    return [w for w, _ in freq.most_common(top_n)]


def parse_titled(source_file, marker):
    items = []
    for obj in load_jsonl(DFIR_DIR / source_file):
        text = obj.get("text", "")
        if marker not in text or len(text) < 50:
            continue
        lines = text.split("\n")
        title = lines[0].replace(marker, "").strip()
        body = "\n".join(lines[1:]).strip()
        if len(body) >= 30:
            items.append({"title": title, "body": body})
    return items


def main():
    rng = random.Random(SEED)
    print("=" * 60)
    print("SecGPT-Prod — Build Evaluation Set")
    print("=" * 60)

    print(f"\n  Loading training data for leakage check: {TRAIN_FILE}")
    train_instructions = []
    train_responses = set()
    with open(TRAIN_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                train_instructions.append(o["instruction"])
                train_responses.add(o["response"][:600])
    train_blob = "\n".join(train_instructions)
    print(f"  Training pairs indexed: {len(train_instructions):,}")

    leaked_count = [0]

    def is_leaked(body, needle=None):
        if body[:600] in train_responses:
            return True
        if needle and needle in train_blob:
            return True
        return False

    eval_items = []

    def add(category, instruction, meta):
        eval_items.append({"id": len(eval_items), "category": category,
                           "instruction": instruction, "meta": meta})

    print("\n  [ttp] MITRE ATT&CK ...")
    mitre = []
    for obj in load_jsonl(DFIR_DIR / "mitre_attack.jsonl"):
        text = obj.get("text", "")
        if len(text) < 80:
            continue
        lines = text.split("\n")
        title = lines[0].replace("MITRE ATT&CK", "").strip().lstrip(":").strip()
        body = "\n".join(lines[1:]).strip()
        if len(body) >= 50 and ("Technique:" in text or "ID: T" in text):
            mitre.append({"title": title, "body": body, "ids": TECH_ID_RE.findall(text)})
    rng.shuffle(mitre)
    n = 0
    for it in mitre:
        if n >= 30:
            break
        if not it["ids"]:
            continue
        leaked = is_leaked(it["body"], it["title"])
        leaked_count[0] += leaked
        if it["title"].lower().startswith("mitigation:"):
            short = it["title"].split(":", 1)[1].strip()
            q = f"How can defenders mitigate {short}?"
        elif n % 3 == 0:
            q = f"Explain MITRE ATT&CK technique {it['ids'][0]}."
        else:
            q = f"What is {it['title']} and how do adversaries use it?"
        add("ttp", q,
            {"title": it["title"], "ids": it["ids"], "keywords": keywords(it["body"]), "leaked": leaked})
        n += 1
    print(f"      {n} prompts")

    print("  [ttp] CAPEC + CISA KEV ...")
    for src, marker, tmpl, cap in [
        ("capec.jsonl", "CAPEC Attack Pattern:", "Explain the attack pattern: {}", 10),
        ("cisa_kev.jsonl", "CISA Known Exploited Vulnerability:", "What is {} and how is it exploited?", 10),
    ]:
        pool = parse_titled(src, marker)
        rng.shuffle(pool)
        n = 0
        for it in pool:
            if n >= cap:
                break
            leaked = is_leaked(it["body"], it["title"])
            leaked_count[0] += leaked
            add("ttp", tmpl.format(it["title"]),
                {"title": it["title"], "ids": [], "keywords": keywords(it["body"]), "leaked": leaked})
            n += 1
        print(f"      {src}: {n}")

    print("  [rule] Sigma / Elastic / Splunk / Atomic ...")
    for src, marker, tmpl, cap, kind in [
        ("sigma.jsonl", "Sigma Detection Rule:", "Write a Sigma detection rule for: {}", 25, "yaml"),
        ("elastic.jsonl", "Elastic Detection Rule:", "Write an Elastic detection rule for: {}", 10, "kql"),
        ("splunk_security.jsonl", "Splunk Detection:", "Write a Splunk detection for: {}", 10, "spl"),
        ("atomic.jsonl", "Atomic Red Team Test:", "How would you test for: {}", 5, "prose"),
    ]:
        pool = parse_titled(src, marker)
        rng.shuffle(pool)
        n = 0
        for it in pool:
            if n >= cap:
                break
            leaked = is_leaked(it["body"], it["title"])
            leaked_count[0] += leaked
            add("rule", tmpl.format(it["title"]),
                {"title": it["title"], "kind": kind, "keywords": keywords(it["body"]), "leaked": leaked})
            n += 1
        print(f"      {src}: {n}")

    print("  [ref] LOLBAS / GTFOBins / Forensics / KAPE / Velociraptor ...")
    for src, marker, tmpl, cap in [
        ("lolbas.jsonl", "LOLBAS:", "How can {} be abused in a living-off-the-land attack?", 20),
        ("gtfobins.jsonl", "GTFOBins:", "How can {} be used to bypass security restrictions on Linux?", 10),
        ("forensic_artifacts.jsonl", "Forensic Artifact:", "What is the {} forensic artifact and how is it useful in investigations?", 10),
        ("kape.jsonl", "KAPE Target:", "What does the KAPE target {} collect and why?", 5),
        ("velociraptor.jsonl", "Velociraptor Artifact:", "What does the Velociraptor artifact {} do?", 5),
    ]:
        pool = parse_titled(src, marker)
        rng.shuffle(pool)
        n = 0
        for it in pool:
            if n >= cap:
                break
            leaked = is_leaked(it["body"], it["title"])
            leaked_count[0] += leaked
            add("ref", tmpl.format(it["title"]),
                {"title": it["title"], "keywords": keywords(it["body"]), "leaked": leaked})
            n += 1
        print(f"      {src}: {n}")

    print("  [kb] CADRE KB chunks ...")
    if CADRE_KB.exists():
        collections_wanted = {
            "HTB Academy", "SANS", "Malpedia", "Offensive Security", "APTnotes",
            "eLearnSecurity", "Mandiant Enterprise IR", "Kaspersky",
            "FLARE Learning Hub", "CISSP",
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
                if 100 <= len(text) <= 1200 and obj.get("title"):
                    candidates.append({"title": obj["title"], "text": text})
        rng.shuffle(candidates)
        n = 0
        for it in candidates:
            if n >= 40:
                break
            if is_leaked(it["text"], it["title"]):
                leaked_count[0] += 1
                continue
            add("kb", f"What is {it['title']} and why is it important in cybersecurity?",
                {"title": it["title"], "keywords": keywords(it["text"]), "leaked": False})
            n += 1
        print(f"      {n} prompts")

    print("  [classification] SMS spam/ham ...")
    sms_zip = DATA / "sms+spam+collection.zip"
    n = 0
    if sms_zip.exists():
        with zipfile.ZipFile(sms_zip) as z:
            with z.open("SMSSpamCollection") as f:
                lines = f.read().decode("latin-1").splitlines()
        rng.shuffle(lines)
        for raw in lines:
            if n >= 25:
                break
            parts = raw.split("\t", 1)
            if len(parts) != 2:
                continue
            label, msg = parts[0].strip(), parts[1].strip()
            if label not in ("spam", "ham") or len(msg) < 10:
                continue
            if msg in train_blob:
                leaked_count[0] += 1
                continue
            add("classification",
                "Classify this SMS message as spam or ham. Explain why.\nMessage: " + msg,
                {"label": label, "leaked": False})
            n += 1
    print(f"      {n} prompts")

    print("  [classification] NSL-KDD connections ...")
    kdd_file = DATA / "KDD+.txt"
    n = 0
    n_normal = 0
    if kdd_file.exists():
        lines = kdd_file.read_text(encoding="utf-8", errors="replace").splitlines()
        rng.shuffle(lines)
        n = 0
        for rec in lines:
            if n >= 25:
                break
            parts = rec.strip().split(",")
            if len(parts) < 42:
                continue
            label = parts[-2].strip().rstrip(".")
            is_normal = label == "normal"
            if is_normal and n_normal >= 8:
                continue
            if rec in train_blob:
                leaked_count[0] += 1
                continue
            add("classification",
                f"Analyze this network connection and classify it.\nConnection: {rec}",
                {"label": label, "leaked": False})
            n += 1
            n_normal += 1 if is_normal else 0
    print(f"      {n} prompts")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for item in eval_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    counts = Counter(i["category"] for i in eval_items)
    held_out = sum(1 for i in eval_items if not i["meta"].get("leaked"))
    print(f"\n{'=' * 60}")
    print(f"  Eval set: {len(eval_items)} prompts -> {OUT_FILE}")
    for cat, c in sorted(counts.items()):
        print(f"    {cat:15s} {c}")
    print(f"  Held-out (generalization): {held_out}")
    print(f"  In-training (recall):      {len(eval_items) - held_out}  ({leaked_count[0]} leaked candidates seen)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
