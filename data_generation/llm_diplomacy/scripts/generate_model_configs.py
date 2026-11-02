#!/usr/bin/env python3
"""Generate deterministic per-power model assignments for AI_Diplomacy.

This is a portable, security-conscious reconstruction of generate_configs.py
in the supplied llm_dip.zip. It does not reproduce the unseeded historical draw.
"""
import argparse
import random
from pathlib import Path

POWERS = ("AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY")


def model_spec(model_id: str, url: str) -> str:
    """Use the upstream OpenAI-compatible client for a local vLLM endpoint."""
    if not model_id.strip() or any(ch in model_id for ch in ["#", "@", ",", "\n"]):
        raise ValueError("Model id must be nonempty and contain no #, @, commas or newlines")
    if not url.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise ValueError("This example is limited to localhost model endpoints")
    if any(ch in url for ch in ["#", ",", "\n", " "]):
        raise ValueError("URL contains unsupported characters")
    return f"openai:{model_id}@{url.rstrip('/')}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llama-id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--qwen-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--llama-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--qwen-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--output", type=Path, default=Path("model_configs_500.txt"))
    args = parser.parse_args()
    if not 1 <= args.games <= 100000:
        parser.error("--games must be between 1 and 100000")
    try:
        models = [model_spec(args.llama_id, args.llama_url), model_spec(args.qwen_id, args.qwen_url)]
    except ValueError as exc:
        parser.error(str(exc))
    if args.output.exists():
        parser.error(f"Output already exists; choose a new path: {args.output}")
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for _ in range(args.games):
            handle.write(",".join(rng.choices(models, k=len(POWERS))) + "\n")
    print(f"Wrote {args.games} game configurations to {args.output}")
    print(f"Seed={args.seed}; power order={', '.join(POWERS)}")


if __name__ == "__main__":
    main()
