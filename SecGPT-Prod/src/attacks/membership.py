"""
membership.py — OWASP ML04 Membership Inference, ATLAS AML.T0022.

Uses the eval set's built-in leakage flags: items whose source text is in
sft_v3.jsonl (meta.leaked=True) vs held-out items. If the model assigns
higher confidence (mean token log-prob of its own greedy response) to
leaked items, membership is inferable.

Metric: mean log-prob gap + AUROC.

Usage:
  python src/attacks/membership.py [--n 40] [--models qwen,gpt2,v25]
"""

import argparse
import io
import json
import sys
from pathlib import Path

import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import eval as harness  # noqa: E402

PROD = Path(__file__).resolve().parent.parent


@torch.no_grad()
def response_logprob(handle, prompt, max_new=120):
    kind = handle["kind"]
    model, tok = handle["model"], handle["tokenizer"]
    if kind == "qwen":
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to("cuda")
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        gen = out[0][ids["input_ids"].shape[1]:]
        if len(gen) == 0:
            return None
        logits = model(out).logits[0, ids["input_ids"].shape[1] - 1: -1]
        logp = torch.log_softmax(logits.float(), dim=-1)
        tok_lp = logp[range(len(gen)), gen]
        return tok_lp.mean().item()
    if kind == "gpt2":
        p = f"Question: {prompt}\nAnswer:"
        ids = tok(p, return_tensors="pt").to("cuda")
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        gen = out[0][ids["input_ids"].shape[1]:]
        if len(gen) == 0:
            return None
        logits = model(out).logits[0, ids["input_ids"].shape[1] - 1: -1]
        logp = torch.log_softmax(logits.float(), dim=-1)
        tok_lp = logp[range(len(gen)), gen]
        return tok_lp.mean().item()
    # v25
    p = common.make_prompt_v25(prompt)
    ids = tok.encode(p).ids
    idx = torch.tensor([ids], dtype=torch.long, device="cuda")
    for _ in range(max_new):
        cond = idx[:, -model.config.block_size:]
        logits, _ = model(cond)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat((idx, nxt), dim=1)
    gen = idx[0, len(ids):]
    logits = model(idx)[0][0, len(ids) - 1: -1]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_lp = logp[range(len(gen)), gen]
    return tok_lp.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="qwen,gpt2,v25")
    parser.add_argument("--n", type=int, default=40)
    args = parser.parse_args()

    items = harness.load_sets("eval")
    leaked = [i for i in items if i["meta"].get("leaked")]
    held = [i for i in items if not i["meta"].get("leaked")]
    print(f"eval set: {len(leaked)} leaked / {len(held)} held-out")

    results = {}
    for name in args.models.split(","):
        handle = common.load_model(name)
        print(f"\n=== {name} ===")
        scores = {"leaked": [], "held_out": []}
        for grp, pool in (("leaked", leaked), ("held_out", held)):
            import random
            random.Random(42).shuffle(pool)
            for it in pool[: args.n]:
                lp = response_logprob(handle, it["instruction"])
                if lp is not None:
                    scores[grp].append(lp)
            print(f"  {grp}: n={len(scores[grp])}, mean lp={sum(scores[grp])/max(len(scores[grp]),1):.3f}")

        l, h = scores["leaked"], scores["held_out"]
        n = min(len(l), len(h))
        if n == 0:
            results[name] = {"gap": None, "auroc": None, "n_leaked": len(l), "n_held": len(h)}
            continue
        # AUROC: P(leaked score > held score) over all pairs
        import itertools
        pairs = list(itertools.product(l[:n], h[:n]))
        auroc = sum(1 for a, b in pairs if a > b) / len(pairs)
        gap = (sum(l) / len(l)) - (sum(h) / len(h))
        print(f"  confidence gap (leaked - held): {gap:+.3f} | AUROC {auroc:.3f}")
        results[name] = {"gap": round(gap, 4), "auroc": round(auroc, 4),
                         "mean_lp_leaked": round(sum(l)/len(l), 4),
                         "mean_lp_held": round(sum(h)/len(h), 4),
                         "n_leaked": len(l), "n_held": len(h)}

    path = common.save_results("membership", results,
                               {"atlas": "AML.T0022", "owasp_ml": "ML04",
                                "method": "mean token log-prob of greedy response",
                                "n_per_group": args.n})
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
