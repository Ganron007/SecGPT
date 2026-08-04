"""
eval.py — SecGPT-Prod benchmark runner.

Runs a model over eval/eval_set.jsonl, scores responses with objective
rule-based scorers, and writes eval/results/<name>_<timestamp>.json.

Scorers (no LLM judge):
  ttp            MITRE ID correctness + hallucination rate + keyword coverage
  rule           YAML/Sigma schema validity (sigma kind) + keyword coverage
  ref            entity mention + keyword coverage
  kb             keyword coverage
  classification label accuracy (spam/ham, normal/attack)

Split metrics: held-out (generalization) vs in-training (recall).

Usage:
  python src/eval.py                                  # SFT checkpoint-500
  python src/eval.py --lora stage1_sft/output/qwen_qlora/checkpoint-500 --name sft500
  python src/eval.py --model openai-community/gpt2 --no-lora --name gpt2-base
  python src/eval.py --limit 2                        # smoke test (2 per category)
  python src/eval.py --compare results/a.json results/b.json
"""

import argparse
import io
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
EVAL_SET = ROOT / "eval" / "eval_set.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
DEFAULT_LORA = ROOT / "stage1_sft" / "output" / "qwen_qlora" / "checkpoint-500"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

TECH_ID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
MAX_NEW_TOKENS = {"classification": 100, "kb": 300, "ttp": 300, "ref": 300,
                  "rule": 400, "soc_triage": 400, "ttp_extract": 300,
                  "forensic_interp": 300, "rule_from_scenario": 400,
                  "consistency": 300}


def kw_score(response, kws):
    if not kws:
        return 1.0
    low = response.lower()
    return sum(1 for k in kws if k in low) / len(kws)


def score_ttp(resp, meta):
    ids_found = list(dict.fromkeys(TECH_ID_RE.findall(resp)))
    correct = [i for i in ids_found if i in meta.get("ids", [])]
    hallucinated = [i for i in ids_found if i not in meta.get("ids", [])]
    kw = kw_score(resp, meta.get("keywords", []))
    id_ok = not meta.get("ids") or bool(correct) or not ids_found
    passed = kw >= 0.5 and id_ok
    return passed, {"kw": round(kw, 2), "ids_found": ids_found, "correct": correct,
                    "hallucinated": hallucinated}


def extract_yaml(resp):
    import yaml
    candidates = [resp]
    if "```" in resp:
        for block in re.findall(r"```(?:yaml|yml)?\n(.*?)```", resp, re.DOTALL):
            candidates.insert(0, block)
    m = re.search(r"(?m)^(title|name):.*$", resp)
    if m:
        candidates.insert(0, resp[m.start():])
    for cand in candidates:
        try:
            doc = yaml.safe_load(cand)
        except Exception:
            continue
        if isinstance(doc, dict):
            return doc
    return None


def score_rule(resp, meta):
    kw = kw_score(resp, meta.get("keywords", []))
    detail = {"kw": round(kw, 2), "yaml_valid": False, "schema_ok": False, "flat_ok": False}
    doc = extract_yaml(resp) if meta.get("kind") == "yaml" else None
    if doc:
        detail["yaml_valid"] = True
        det = doc.get("detection")
        if isinstance(det, dict):
            condition = str(det.get("condition", ""))
            selections = [k for k in det if k != "condition"]
            detail["schema_ok"] = bool(condition) and any(s in condition for s in selections)
    cond = re.search(r"(?im)^\s*condition:\s*\S", resp)
    detail["flat_ok"] = bool(cond) and "selection" in resp
    marker = any(m in resp for m in ("condition", "selection", "|", "filter", "EventID"))
    passed = detail["schema_ok"] or detail["flat_ok"] or (kw >= 0.5 and marker)
    return passed, detail


def score_ref(resp, meta):
    title = meta.get("title", "")
    low = resp.lower()
    base = re.sub(r"\.\w{2,4}$", "", title.lower())
    mentioned = title.lower() in low or (len(base) > 3 and base in low)
    kw = kw_score(resp, meta.get("keywords", []))
    return mentioned and kw >= 0.375, {"kw": round(kw, 2), "mentioned": mentioned}


def score_kb(resp, meta):
    kw = kw_score(resp, meta.get("keywords", []))
    return kw >= 0.5, {"kw": round(kw, 2)}


def score_classification(resp, meta):
    label = meta["label"].lower()
    low = resp.lower()
    head = low[:120]
    if label in ("spam", "ham"):
        spam_pos = head.find("spam")
        ham_pos = head.find("ham")
        pred = "spam" if spam_pos != -1 and (ham_pos == -1 or spam_pos < ham_pos) else "ham"
        passed = pred == label
    elif label == "normal":
        passed = "normal" in head and "attack" not in head[:80]
    else:
        passed = label in low or ("attack" in head and "normal" not in head[:80])
    return passed, {"label": label}


SEVERITY_WORDS = ("critical", "high", "medium", "low", "informational")
VERDICT_WORDS = ("true positive", "false positive", "malicious", "suspicious", "benign", "legitimate")
ACTION_WORDS = ("investigate", "isolate", "block", "quarantine", "monitor",
                "escalate", "collect", "contain", "remediate", "disable")


def score_soc_triage(resp, meta):
    low = resp.lower()
    groups = {"severity": any(w in low for w in SEVERITY_WORDS),
              "verdict": any(w in low for w in VERDICT_WORDS),
              "action": any(w in low for w in ACTION_WORDS)}
    hits = sum(groups.values())
    return hits >= 2, {**groups, "hits": hits}


def score_ttp_extract(resp, meta):
    source = meta.get("ids", [])
    found = list(dict.fromkeys(TECH_ID_RE.findall(resp)))
    overlap = [i for i in found if i in source]
    extra = [i for i in found if i not in source]
    recall = len(overlap) / max(len(source), 1)
    return recall >= 0.5 and len(extra) <= 1, {"recall": round(recall, 2), "found": found, "extra": extra}


def score_forensic_interp(resp, meta):
    kw = kw_score(resp, meta.get("keywords", []))
    return kw >= 0.5, {"kw": round(kw, 2)}


CONS_STOP = set("this that with from they them their have been were will would could "
                "should what when where which your about into over under through during".split())


def content_tokens(text):
    return set(w for w in re.findall(r"[a-z][a-z0-9_-]{3,}", text.lower()) if w not in CONS_STOP)


def score_consistency(resps, meta):
    sets = [content_tokens(r) for r in resps]
    pairs = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            smaller = min(len(sets[i]), len(sets[j]))
            pairs.append(len(sets[i] & sets[j]) / max(smaller, 1))
    avg = sum(pairs) / max(len(pairs), 1)
    return avg >= 0.4, {"agreement": round(avg, 2)}


SCORERS = {"ttp": score_ttp, "rule": score_rule, "ref": score_ref,
           "kb": score_kb, "classification": score_classification,
           "soc_triage": score_soc_triage, "ttp_extract": score_ttp_extract,
           "forensic_interp": score_forensic_interp, "rule_from_scenario": score_rule}


PRACTICAL_SET = ROOT / "eval" / "practical_set.jsonl"


def load_sets(which, limit=None):
    files = {"eval": [EVAL_SET], "practical": [PRACTICAL_SET],
             "all": [EVAL_SET, PRACTICAL_SET]}[which]
    items = []
    for fp in files:
        if not fp.exists():
            continue
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
    if limit:
        per_cat = {}
        limited = []
        for it in items:
            n = per_cat.get(it["category"], 0)
            if n < limit:
                limited.append(it)
                per_cat[it["category"]] = n + 1
        return limited
    return items


def run_generation(items, model_name, lora_path, batch_size):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    t0 = time.time()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
    if lora_path:
        from peft import PeftModel
        print(f"  Loading LoRA: {lora_path}")
        model = PeftModel.from_pretrained(model, str(lora_path))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    load_s = time.time() - t0

    responses = {}
    total_new_tokens = 0
    t_gen = time.time()
    cats = {}
    for it in items:
        cats.setdefault(it["category"], []).append(it)
    for cat, cat_items in cats.items():
        max_new = MAX_NEW_TOKENS.get(cat, 300)
        tasks = []
        for b in cat_items:
            if cat == "consistency":
                for idx, ph in enumerate(b["meta"]["phrasings"]):
                    tasks.append((b["id"], idx, ph))
            else:
                tasks.append((b["id"], 0, b["instruction"]))
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            texts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": t[2]}],
                tokenize=False, add_generation_prompt=True) for t in batch]
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            for j, (item_id, idx, _) in enumerate(batch):
                new_ids = out[j][inputs["input_ids"].shape[1]:]
                total_new_tokens += len(new_ids)
                responses.setdefault(item_id, {})[idx] = \
                    tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            print(f"    [{cat}] {min(i + batch_size, len(tasks))}/{len(tasks)}", flush=True)
    gen_s = time.time() - t_gen
    vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    return responses, {"load_s": round(load_s, 1), "gen_s": round(gen_s, 1),
                       "new_tokens": total_new_tokens,
                       "tok_per_s": round(total_new_tokens / max(gen_s, 0.1), 1),
                       "peak_vram_gb": round(vram, 2)}


def compute_metrics(scored):
    cats = {}
    for it in scored:
        cats.setdefault(it["category"], []).append(it)
    per_cat = {}
    for cat, items in cats.items():
        n = len(items)
        passes = sum(1 for i in items if i["pass"])
        entry = {"n": n, "pass_rate": round(passes / n, 3)}
        if cat == "ttp":
            with_ids = [i for i in items if i["detail"].get("ids_found")]
            halluc = [i for i in with_ids if i["detail"].get("hallucinated")]
            entry["hallucination_rate"] = round(len(halluc) / max(len(with_ids), 1), 3)
        if cat == "rule":
            sigma = [i for i in items if i["detail"].get("yaml_valid") is not None]
            entry["yaml_valid_rate"] = round(sum(1 for i in sigma if i["detail"]["yaml_valid"]) / max(len(sigma), 1), 3)
            entry["schema_ok_rate"] = round(sum(1 for i in sigma if i["detail"]["schema_ok"]) / max(len(sigma), 1), 3)
        per_cat[cat] = entry
    held = [i for i in scored if not i["meta"].get("leaked")]
    recall = [i for i in scored if i["meta"].get("leaked")]
    overall = {
        "n": len(scored),
        "pass_rate": round(sum(1 for i in scored if i["pass"]) / max(len(scored), 1), 3),
        "held_out": {"n": len(held),
                     "pass_rate": round(sum(1 for i in held if i["pass"]) / max(len(held), 1), 3)},
        "recall": {"n": len(recall),
                   "pass_rate": round(sum(1 for i in recall if i["pass"]) / max(len(recall), 1), 3)},
    }
    return {"overall": overall, "per_category": per_cat}


def print_report(name, metrics):
    print(f"\n{'=' * 64}")
    print(f"  RESULTS: {name}")
    print(f"{'=' * 64}")
    o = metrics["overall"]
    print(f"  Overall pass rate:   {o['pass_rate']:.1%}  ({o['n']} prompts)")
    print(f"  Held-out (general):  {o['held_out']['pass_rate']:.1%}  ({o['held_out']['n']})")
    print(f"  Recall (trained):    {o['recall']['pass_rate']:.1%}  ({o['recall']['n']})")
    print(f"\n  {'Category':15s} {'n':>4s} {'pass':>7s} {'extra':>30s}")
    for cat, e in sorted(metrics["per_category"].items()):
        extra = ""
        if "hallucination_rate" in e:
            extra = f"hallucination {e['hallucination_rate']:.1%}"
        if "yaml_valid_rate" in e:
            extra = f"yaml {e['yaml_valid_rate']:.1%} / schema {e['schema_ok_rate']:.1%}"
        print(f"  {cat:15s} {e['n']:>4d} {e['pass_rate']:>7.1%} {extra:>30s}")


def compare(paths):
    results = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    names = [r["name"] for r in results]
    width = max(12, max(len(n) for n in names) + 2)
    print(f"\n{'=' * (24 + width * len(results))}")
    print(f"  COMPARE: {'  vs  '.join(names)}")
    print(f"{'=' * (24 + width * len(results))}")

    def row(label, values):
        cells = "".join(f"{v:>{width}.1%}" if v is not None else f"{'—':>{width}}" for v in values)
        print(f"  {label:20s}{cells}")

    cats = sorted({c for r in results for c in r["metrics"]["per_category"]})
    print(f"\n  {'':20s}" + "".join(f"{n:>{width}s}" for n in names))
    for cat in cats:
        row(cat, [r["metrics"]["per_category"].get(cat, {}).get("pass_rate") for r in results])
    print()
    row("OVERALL", [r["metrics"]["overall"]["pass_rate"] for r in results])
    row("HELD-OUT", [r["metrics"]["overall"]["held_out"]["pass_rate"] for r in results])
    row("RECALL", [r["metrics"]["overall"]["recall"]["pass_rate"] for r in results])
    row("TTP halluc.", [r["metrics"]["per_category"].get("ttp", {}).get("hallucination_rate") for r in results])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--lora", default=str(DEFAULT_LORA))
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--compare", nargs="+", metavar="RESULT", help="compare 2+ result JSONs")
    parser.add_argument("--set", choices=["eval", "practical", "all"], default="all")
    args = parser.parse_args()

    if args.compare:
        compare(args.compare)
        return

    lora = None if args.no_lora else args.lora
    name = args.name or (Path(lora).name if lora else Path(args.model).name)

    print("=" * 64)
    print(f"  SecGPT-Prod Benchmark — {name}")
    print("=" * 64)
    items = load_sets(args.set, args.limit)
    print(f"\n  Eval set: {len(items)} prompts (model={args.model}, lora={lora})")

    responses, perf = run_generation(items, args.model, lora, args.batch)

    scored = []
    for it in items:
        resp_map = responses[it["id"]]
        if it["category"] == "consistency":
            resps = [resp_map[i] for i in sorted(resp_map)]
            passed, detail = score_consistency(resps, it["meta"])
            scored.append({**it, "response": resps, "pass": passed, "detail": detail})
        else:
            resp = resp_map[0]
            passed, detail = SCORERS[it["category"]](resp, it["meta"])
            scored.append({**it, "response": resp, "pass": passed, "detail": detail})

    metrics = compute_metrics(scored)
    print_report(name, metrics)
    print(f"\n  Perf: load {perf['load_s']}s | gen {perf['gen_s']}s | "
          f"{perf['tok_per_s']} tok/s | peak VRAM {perf['peak_vram_gb']} GB")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out = RESULTS_DIR / f"{name}_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"name": name, "model": args.model, "lora": lora,
                   "timestamp": ts, "perf": perf, "metrics": metrics,
                   "items": scored}, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
