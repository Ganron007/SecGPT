"""
build_practical_set.py — SecGPT-Prod: build the practical-utility benchmark layer.

Scenario tasks (not Q&A) that test whether the model is useful in practice:
  soc_triage        15  triage an alert: verdict + severity + actions
  ttp_extract       10  extract MITRE technique IDs from a threat text
  forensic_interp   10  interpret a forensic artifact in an investigation
  rule_from_scenario 10 write a detection rule from observed behavior
  consistency        6  same question, 3 phrasings -> answer agreement

Output: eval/practical_set.jsonl  (gitignored via *.jsonl)

Usage:
  python src/build_practical_set.py
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
DATA = ROOT.parent / "SecGPTv2" / "data"
DFIR_DIR = DATA / "dfir_nexus_sources"
TRAIN_FILE = ROOT / "data" / "sft_32k.jsonl"
OUT_FILE = ROOT / "eval" / "practical_set.jsonl"

SEED = 42
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
    print("SecGPT-Prod — Build Practical Scenario Set")
    print("=" * 60)

    train_responses = set()
    train_instructions = []
    with open(TRAIN_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                train_instructions.append(o["instruction"])
                train_responses.add(o["response"][:600])
    train_blob = "\n".join(train_instructions)

    def is_leaked(body, needle=None):
        return body[:600] in train_responses or (needle and needle in train_blob)

    items = []

    def add(category, instruction, meta):
        meta["leaked"] = bool(meta.get("leaked", False))
        items.append({"id": 10000 + len(items), "category": category,
                      "instruction": instruction, "meta": meta})

    print("\n  [soc_triage] elastic + sigma alerts ...")
    pool = parse_titled("elastic.jsonl", "Elastic Detection Rule:") + \
           parse_titled("sigma.jsonl", "Sigma Detection Rule:")
    rng.shuffle(pool)
    for it in pool[:15]:
        add("soc_triage",
            f"You are a SOC analyst. Triage this alert: give a verdict, a severity, and recommended actions.\n"
            f"Alert: {it['title']}\nDetails: {it['body'][:400]}",
            {"title": it["title"], "leaked": is_leaked(it["body"], it["title"])})
    print(f"      {min(15, len(pool))} scenarios")

    print("  [ttp_extract] MITRE texts with multiple techniques ...")
    pool = []
    for obj in load_jsonl(DFIR_DIR / "mitre_attack.jsonl"):
        text = obj.get("text", "")
        ids = list(dict.fromkeys(TECH_ID_RE.findall(text)))
        if len(text) >= 200 and len(ids) >= 2:
            pool.append({"body": text.strip(), "ids": ids})
    rng.shuffle(pool)
    for it in pool[:10]:
        add("ttp_extract",
            f"List every MITRE ATT&CK technique ID mentioned in this text, with one line per technique.\n"
            f"Text: {it['body'][:500]}",
            {"ids": it["ids"], "leaked": is_leaked(it["body"])})
    print(f"      {min(10, len(pool))} scenarios")

    print("  [forensic_interp] artifacts + KAPE ...")
    pool = parse_titled("forensic_artifacts.jsonl", "Forensic Artifact:") + \
           parse_titled("kape.jsonl", "KAPE Target:")
    rng.shuffle(pool)
    for it in pool[:10]:
        add("forensic_interp",
            f"During an investigation you found this artifact. Explain what it proves and how you would use it.\n"
            f"Artifact: {it['title']}",
            {"title": it["title"], "keywords": keywords(it["body"]),
             "leaked": is_leaked(it["body"], it["title"])})
    print(f"      {min(10, len(pool))} scenarios")

    print("  [rule_from_scenario] atomic behaviors ...")
    pool = parse_titled("atomic.jsonl", "Atomic Red Team Test:")
    rng.shuffle(pool)
    for it in pool[:10]:
        add("rule_from_scenario",
            f"An adversary was observed performing the following behavior. Write a detection rule to catch it.\n"
            f"Observed behavior: {it['body'][:400]}",
            {"title": it["title"], "kind": "prose", "keywords": keywords(it["body"]),
             "leaked": is_leaked(it["body"], it["title"])})
    print(f"      {min(10, len(pool))} scenarios")

    print("  [consistency] 3 phrasings of the same question ...")
    kb_pool = []
    cadre = DATA / "cadre_kb.jsonl"
    if cadre.exists():
        with open(cadre, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("collection") in ("HTB Academy", "CISSP", "SANS") and obj.get("title"):
                    text = obj.get("text", "")
                    if 100 <= len(text) <= 1200 and text not in train_blob:
                        kb_pool.append(obj["title"])
    rng.shuffle(kb_pool)
    n = 0
    for title in kb_pool:
        if n >= 6:
            break
        if title in train_blob:
            continue
        add("consistency", title,
            {"phrasings": [f"What is {title}?",
                           f"Explain {title} in simple terms.",
                           f"Give me a detailed overview of {title}."],
             "leaked": False})
        n += 1
    print(f"      {n} scenarios")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    counts = Counter(i["category"] for i in items)
    print(f"\n{'=' * 60}")
    print(f"  Practical set: {len(items)} scenarios -> {OUT_FILE}")
    for cat, c in sorted(counts.items()):
        print(f"    {cat:20s} {c}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
