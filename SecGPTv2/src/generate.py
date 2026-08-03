"""
generate.py — SecGPT v2 Step 6: Generate text from trained model.

Usage:
  python src/generate.py --tag kb
  python src/generate.py --tag rule --max-tokens 200
  python src/generate.py --prompt "<|rule|>\nrule Emotet"
  python src/generate.py --interactive
"""

import argparse
import io
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model import GPT, GPTConfig

STEP5_OUT = ROOT / "stage1_pre-training" / "step5_training" / "output"
STEP1_OUT = ROOT / "stage1_pre-training" / "step1_tokenizer" / "output"

TAGS = ["kb", "rule", "ttp", "ref", "spam", "ham", "net", "malware"]


def load_model(checkpoint_path=None, device="cuda"):
    ckpt_path = Path(checkpoint_path) if checkpoint_path else STEP5_OUT / "checkpoint_final.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config, ckpt.get("step", "?")


def load_tokenizer():
    return Tokenizer.from_file(str(STEP1_OUT / "tokenizer.json"))


def generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, device="cuda"):
    encoded = tokenizer.encode(prompt)
    idx = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
    return tokenizer.decode(out[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="SecGPT v2 Generation")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, step = load_model(args.checkpoint, device)
    tokenizer = load_tokenizer()

    print(f"SecGPT v2 | {model.num_parameters()/1e6:.1f}M params | step {step} | {device}")

    if args.interactive:
        print(f"\nInteractive mode. Type prompts, /temp X, /len X, /quit")
        print("-" * 50)
        temp, length = args.temperature, args.max_tokens
        while True:
            try:
                user_input = input("\nprompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                continue
            if user_input == "/quit":
                break
            if user_input.startswith("/temp "):
                temp = float(user_input.split()[1])
                print(f"  temp={temp}")
                continue
            if user_input.startswith("/len "):
                length = int(user_input.split()[1])
                print(f"  len={length}")
                continue
            prompt = user_input.replace("\\n", "\n")
            text = generate_text(model, tokenizer, prompt, length, temp, device)
            body = text[len(prompt):]
            print(f"\n{body}")
        return

    if args.prompt is not None:
        prompt = args.prompt.replace("\\n", "\n")
        text = generate_text(model, tokenizer, prompt, args.max_tokens, args.temperature, device)
        print(f"\n{text}")
        return

    tags_to_run = [args.tag] if args.tag else TAGS
    for tag in tags_to_run:
        prompt = f"<|{tag}|>\n"
        print(f"\n{'=' * 50}")
        print(f"  <|{tag}|>")
        print(f"{'=' * 50}")
        for i in range(args.samples):
            text = generate_text(model, tokenizer, prompt, args.max_tokens, args.temperature, device)
            body = text[len(prompt):]
            print(f"\n[{i+1}] {body}")


if __name__ == "__main__":
    main()
