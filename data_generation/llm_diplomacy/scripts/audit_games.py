#!/usr/bin/env python3
"""Audit previously generated AI_Diplomacy game files without modifying them.

Usage:
    python audit_games.py --results-dir /path/to/AI_Diplomacy/results \
        --output reports/game_manifest.csv

All statistics are extracted from saved files, NOT from the theoretical game config.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

POWERS = ("AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY")
COLUMNS = (
    "game_id", "phase_count", "last_phase", "model_a", "model_b",
    "order_attempts", "no_moves_extracted", "other_order_outcomes",
    "game_orders_total", "non_hold_orders", "has_overview", "issues",
)


def read_game(run_dir: Path) -> dict:
    problems = []
    result = dict.fromkeys(COLUMNS, "")
    result["game_id"] = run_dir.name
    result["has_overview"] = int((run_dir / "overview.jsonl").is_file())
    game_file = run_dir / "lmvsgame.json"
    phases = []
    if game_file.is_file():
        try:
            game = json.loads(game_file.read_text(encoding="utf-8"))
            phases = game.get("phases") or []
        except (ValueError, OSError) as exc:
            problems.append(f"Invalid game JSON: {type(exc).__name__}")
    else:
        problems.append("Missing lmvsgame.json")

    result["phase_count"] = len(phases)
    result["last_phase"] = phases[-1].get("name", "") if phases else ""
    if phases:
        assignments = phases[0].get("state_agents") or {}
        model_ids = Counter()
        for power in POWERS:
            meta = assignments.get(power) or {}
            raw = meta.get("model_id") or ""
            if raw:
                # Do not record model endpoints or inlined credentials.
                model_ids[str(raw).split("@")[0].split("#")[0]] += 1
        models = sorted(model_ids)
        result["model_a"] = models[0] if models else ""
        result["model_b"] = models[1] if len(models) > 1 else ""
    n_orders = 0
    n_nonholds = 0
    for phase in phases:
        for orders in (phase.get("orders") or {}).values():
            for item in (orders or []):
                n_orders += 1
                if not (isinstance(item, str) and item.strip().endswith(" H")):
                    n_nonholds += 1
    result["game_orders_total"] = n_orders
    result["non_hold_orders"] = n_nonholds

    log_file = run_dir / "llm_responses.csv"
    counts = Counter()
    if log_file.is_file():
        with log_file.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for record in reader:
                if record.get("response_type") == "order_generation":
                    counts[record.get("success", "")] += 1
    else:
        problems.append("Missing llm_responses.csv")
    n_attempts = sum(counts.values())
    no_moves = counts["Failure: No moves extracted"]
    result["order_attempts"] = n_attempts
    result["no_moves_extracted"] = no_moves
    result["other_order_outcomes"] = n_attempts - no_moves
    if n_attempts and no_moves == n_attempts:
        problems.append("All order responses unparseable; fallback used")
    if len(phases) < 7:
        problems.append("Fewer than seven saved phases; check run completion")
    result["issues"] = "; ".join(problems)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.results_dir.is_dir():
        parser.error(f"Not a directory: {args.results_dir}")
    folders = sorted(folder for folder in args.results_dir.glob("game_*") if folder.is_dir())
    if not folders:
        parser.error("No game_* subdirectories found")
    rows = [read_game(folder) for folder in folders]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Games scanned: {len(rows)}")
    print(f"Logged order attempts: {sum(r['order_attempts'] for r in rows):,}")
    print(f"No-moves-extracted failures: {sum(r['no_moves_extracted'] for r in rows):,}")
    print(f"Manifest: {args.output}")


if __name__ == "__main__":
    main()
