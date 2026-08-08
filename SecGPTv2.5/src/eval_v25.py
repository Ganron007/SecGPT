"""
eval_v25.py — SecGPTv2.5 benchmark adapter.

Runs the shared 291-prompt harness on the custom from-scratch model
(own tokenizer, tag-based Q/A prompting, greedy decoding, stop at pair
separator). Scorers/metrics reused from SecGPT-Prod/src/eval.py.

Usage:
  python src/eval_v25.py --checkpoint stage1_pre-training/step5_training/output/checkpoint_final.pt --name v25pre
  python src/eval_v25.py --checkpoint stage2_sft/output/checkpoint_sft.pt --name v25sft
  python src/eval_v25.py --checkpoint stage3_alignment/output/checkpoint_dpo.pt --name v25dpo
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "SecGPT-Prod" / "src"))

from src.model import GPT, GPTConfig  # noqa: E402
import eval as harness  # noqa: E402

TOK_PATH = ROOT / "stage1_pre-training" / "step1_tokenizer" / "output" / "tokenizer.json"
RESULTS_DIR = ROOT.parent / "SecGPT-Prod" / "eval" / "results"
TAG_MAP = {"ttp": "ttp", "rule": "rule", "ref": "ref", "kb": "kb",
           "classification": "kb", "soc_triage": "kb", "ttp_extract": "ttp",
           "forensic_interp": "ref", "rule_from_scenario": "rule",
           "consistency": "kb"}
MAX_NEW = {"classification": 100, "kb": 300, "ttp": 300, "ref": 300, "rule": 400,
           "soc_triage": 400, "ttp_extract": 300, "forensic_interp": 300,
           "rule_from_scenario": 400, "consistency": 300}


@torch.no_grad()
def generate_greedy(model, idx, max_new, stop_ids, block_size):
    for _ in range(max_new):
        cond = idx[:, -block_size:]
        logits, _ = model(cond)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat((idx, nxt), dim=1)
        if stop_ids is not None and idx[0, -1].item() in stop_ids:
            if len(idx[0]) >= 2 and idx[0, -2].item() in stop_ids:
                break
    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 64)
    print(f"SecGPTv2.5 benchmark — {args.name}")
    print("=" * 64)

    tokenizer = Tokenizer.from_file(str(TOK_PATH))
    nl_id = tokenizer.encode("\n").ids[0]

    ckpt = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Loaded {args.checkpoint} ({model.num_parameters()/1e6:.1f}M params)")

    items = harness.load_sets("all", args.limit)
    print(f"  Eval items: {len(items)}")

    t0 = time.time()
    scored = []
    for n, it in enumerate(items):
        cat = it["category"]
        if cat == "consistency":
            resps = []
            for ph in it["meta"]["phrasings"]:
                resps.append(run_one(model, tokenizer, ph, "kb", MAX_NEW["consistency"],
                                     nl_id, config.block_size, device))
            passed, detail = harness.score_consistency(resps, it["meta"])
            scored.append({**it, "response": resps, "pass": passed, "detail": detail})
        else:
            resp = run_one(model, tokenizer, it["instruction"], TAG_MAP.get(cat, "kb"),
                           MAX_NEW.get(cat, 300), nl_id, config.block_size, device)
            passed, detail = harness.SCORERS[cat](resp, it["meta"])
            scored.append({**it, "response": resp, "pass": passed, "detail": detail})
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(items)} ({(time.time()-t0)/60:.1f} min)", flush=True)

    metrics = harness.compute_metrics(scored)
    harness.print_report(args.name, metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M")
    out = RESULTS_DIR / f"{args.name}_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"name": args.name, "model": f"SecGPTv2.5/{args.checkpoint}",
                   "lora": None, "timestamp": ts,
                   "perf": {"gen_min": round((time.time() - t0) / 60, 1)},
                   "metrics": metrics, "items": scored}, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out}")


def run_one(model, tokenizer, instruction, tag, max_new, nl_id, block_size, device):
    prompt = f"<|{tag}|>\nQ: {instruction}\nA:"
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate_greedy(model, idx, max_new, {nl_id}, block_size)
    ans = tokenizer.decode(out[0, len(ids):].tolist())
    for stop in ("\n\nQ:", "\n\n<|", "\nQ:"):
        if stop in ans:
            ans = ans[:ans.index(stop)]
    return ans.strip()


if __name__ == "__main__":
    main()
