#!/usr/bin/env python3
"""Run a small, controllable batch using a separately obtained AI_Diplomacy engine.

Historical settings are documented in docs/HISTORICAL_PROVENANCE.md.
Do not use for dataset creation until the upstream licensing position is clear.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def load_configs(path: Path) -> list[str]:
    if not path.is_file():
        raise ValueError(f"Configuration file not found: {path}")
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Configuration file is empty")
    for n, row in enumerate(rows, start=1):
        if len(row.split(",")) != 7:
            raise ValueError(f"Line {n}: expected seven model specifications")
        if "#" in row or "\n" in row:
            raise ValueError(f"Line {n}: inline keys (#...) are not allowed; use environment variables")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", type=Path, required=True, help="AI_Diplomacy checkout containing lm_game.py")
    parser.add_argument("--config", type=Path, required=True, help="Seven model specs per line")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1, help="Run at most this many configurations (default 1)")
    parser.add_argument("--max-year", type=int, default=1903)
    parser.add_argument("--negotiation-rounds", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=2048, help="Safer exploratory default; original archived batch used 100")
    parser.add_argument("--prompts-dir", type=Path, default=None, help="Optional custom prompts; default uses upstream prompts_simple")
    parser.add_argument("--python", default=sys.executable, help="Interpreter with upstream AI_Diplomacy dependencies")
    parser.add_argument("--dry-run", action="store_true", help="Print commands but never execute")
    args = parser.parse_args()
    if not (args.engine_dir / "lm_game.py").is_file():
        parser.error("--engine-dir must contain lm_game.py")
    if args.limit < 1 or args.start_index < 0 or args.max_tokens < 1:
        parser.error("--limit and --max-tokens must be positive; --start-index must be nonnegative")
    if args.negotiation_rounds < 0 or args.max_year < 1901:
        parser.error("Invalid --negotiation-rounds or --max-year")
    try:
        configs = load_configs(args.config)
    except ValueError as exc:
        parser.error(str(exc))
    if args.start_index >= len(configs):
        parser.error("--start-index exceeds the configuration file length")
    if args.prompts_dir is not None and not args.prompts_dir.is_dir():
        parser.error("--prompts-dir must exist")
    selected = range(args.start_index, min(args.start_index + args.limit, len(configs)))
    for i in selected:
        out_dir = args.output_dir.resolve() / f"game_{i:03d}"
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"SKIP: Nonempty run directory exists: {out_dir}")
            continue
        command = [args.python, "lm_game.py", "--run_dir", str(out_dir),
                   "--max_year", str(args.max_year),
                   "--num_negotiation_rounds", str(args.negotiation_rounds),
                   "--models", configs[i], "--max_tokens", str(args.max_tokens),
                   "--generate_phase_summaries", "false",
                   "--use_unformatted_prompts", "false"]
        if args.prompts_dir:
            command.extend(["--prompts_dir", str(args.prompts_dir.resolve())])
        printable = " ".join(__import__("shlex").quote(arg) for arg in command)
        print(f"GAME {i}: {printable}")
        if not args.dry_run:
            # No shell=True; all model IDs are passed as a single CLI argument.
            subprocess.run(command, cwd=args.engine_dir.resolve(), env=os.environ.copy(), check=True)
    print("Dry run only; no simulations performed." if args.dry_run else "Requested batch finished.")


if __name__ == "__main__":
    main()
