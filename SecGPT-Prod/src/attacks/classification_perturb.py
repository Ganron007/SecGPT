"""
classification_perturb.py — OWASP ML01 Input Manipulation, ATLAS AML.T0017.

Perturbs classifier inputs (SMS text, NSL-KDD feature rows) and measures
accuracy drop on all three models vs clean inputs.

Perturbations:
  sms: 10% of alphabetic characters randomly swapped with a neighbor key
  kdd: flip numeric feature values by +-10% (features 4..40)

Usage:
  python src/attacks/classification_perturb.py [--n 30] [--models qwen,gpt2,v25]
"""

import argparse
import io
import json
import random
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import eval as harness  # noqa: E402


def perturb_sms(text, rng, rate=0.1):
    letters = [i for i, ch in enumerate(text) if ch.isalpha()]
    n = max(1, int(len(letters) * rate))
    picks = rng.sample(letters, min(n, len(letters)))
    out = list(text)
    for i in picks:
        c = out[i]
        out[i] = chr(ord(c) + 1) if c != "z" else "a"
    return "".join(out)


def perturb_kdd(row, rng, rate=0.1):
    parts = row.split(",")
    n = max(1, int((len(parts) - 4) * rate))
    idxs = rng.sample(range(4, len(parts) - 2), min(n, len(parts) - 6))
    for i in idxs:
        try:
            v = float(parts[i])
            parts[i] = str(round(v * rng.uniform(0.9, 1.1), 4))
        except ValueError:
            pass
    return ",".join(parts)


def extract_label(meta):
    return meta.get("label", "").lower()


def classify(resp):
    low = resp.lower()[:150]
    if "spam" in low:
        return "spam"
    if "ham" in low or "legitimate" in low:
        return "ham"
    if "attack" in low or "anomal" in low or "malicious" in low:
        return "attack"
    if "normal" in low:
        return "normal"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="qwen,gpt2,v25")
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()

    items = [i for i in harness.load_sets("eval")
             if i["category"] == "classification"]
    rng = random.Random(42)
    rng.shuffle(items)
    items = items[: args.n]

    results = {}
    for name in args.models.split(","):
        handle = common.load_model(name)
        print(f"\n=== {name} ===")
        clean_ok, perturb_ok, total = 0, 0, 0
        rows = []
        for it in items:
            label = extract_label(it["meta"])
            inst = it["instruction"]
            if "SMS message" in inst:
                msg = re.search(r"Message: (.+)$", inst, re.M).group(1)
                perturbed = perturb_sms(msg, rng)
                p_inst = inst.replace(msg, perturbed)
            else:
                conn = re.search(r"Connection: (.+)$", inst, re.M).group(1)
                perturbed = perturb_kdd(conn, rng)
                p_inst = inst.replace(conn, perturbed)

            c_resp = common.generate(handle, it["instruction"], max_new=120)
            p_resp = common.generate(handle, p_inst, max_new=120)
            c_lab, p_lab = classify(c_resp), classify(p_resp)
            c_ok = c_lab == label
            p_ok = p_lab == label
            clean_ok += c_ok
            perturb_ok += p_ok
            total += 1
            rows.append({"label": label, "clean_ok": c_ok, "perturbed_ok": p_ok,
                         "clean_resp": c_resp[:100], "perturbed_resp": p_resp[:100]})

        clean_acc = clean_ok / total
        perturb_acc = perturb_ok / total
        print(f"  clean acc: {clean_acc:.0%} | perturbed acc: {perturb_acc:.0%} "
              f"| drop: {clean_acc - perturb_acc:+.0%}")
        results[name] = {"clean_accuracy": round(clean_acc, 3),
                         "perturbed_accuracy": round(perturb_acc, 3),
                         "drop": round(clean_acc - perturb_acc, 3),
                         "n": total, "rows": rows}

    path = common.save_results("classification_perturb", results,
                               {"atlas": "AML.T0017", "owasp_ml": "ML01",
                                "perturbations": "sms: char-swap 10%, kdd: feature flip 10%"})
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
