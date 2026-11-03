"""Extract private Diplomacy messages; preserve UTF-8, real phases and file provenance."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from rag_utils.io import atomic_json, file_sha256


def process_game_logs(input_base_dir="llm_dip_results", output_base_dir="game_text", naming="summary"):
    base, output = Path(input_base_dir), Path(output_base_dir)
    if not base.is_dir(): raise FileNotFoundError(base)
    if naming not in {"summary", "original"}: raise ValueError("Invalid naming convention")
    paths = sorted(base.glob("*/lmvsgame.json"))
    if not paths: raise ValueError("No game histories found; demonstration games are not generated")
    output.mkdir(parents=True, exist_ok=True); manifest = []; failures = []
    for p in paths:
        try:
            game = json.loads(p.read_text(encoding="utf-8")); lines = []
            if not isinstance(game.get("phases"), list): raise ValueError("Missing phases list")
            for phase in game["phases"]:
                messages = phase.get("messages", [])
                if not isinstance(messages, list): raise ValueError("Expected phase.messages list")
                for message in messages:
                    if message.get("recipient") == "GLOBAL": continue
                    value = message.get("phase") or phase.get("name")
                    match = re.fullmatch(r"([SFW])(\d{4})[A-Z]*", str(value))
                    if not match: raise ValueError("Missing/invalid message phase; no year is invented")
                    text = message.get("message")
                    if not isinstance(text, str) or not text.strip(): continue
                    sender, recipient = message.get("sender"), message.get("recipient")
                    if not sender or not recipient: raise ValueError("Message missing sender/recipient")
                    season = {"S": "Spring", "F": "Fall", "W": "Winter"}[match.group(1)]
                    lines.append(f"{sender}, {season} {match.group(2)} to {recipient}: {text}")
            if not lines: raise ValueError("Game has no private messages")
            name = p.parent.name + ("_messages.txt" if naming == "summary" else ".txt")
            dest = output / name; content = "\n".join(lines)
            if dest.exists() and dest.read_text(encoding="utf-8") != content:
                raise ValueError("Existing extracted transcript differs; use a new output directory")
            dest.write_text(content, encoding="utf-8")
            manifest.append({"source": str(p.relative_to(base)), "source_sha256": file_sha256(p), "output": name, "private_messages": len(lines), "characters": len(content)})
        except (ValueError, KeyError, TypeError, OSError) as e:
            failures.append({"source": str(p.relative_to(base)), "error_type": type(e).__name__, "reason": str(e)})
    atomic_json(output / "extraction_manifest.json", {"games": manifest, "failures": failures})
    if failures: raise RuntimeError("One or more games could not be extracted; inspect extraction_manifest.json")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--naming", choices=("summary", "original"), default="summary")
    a = p.parse_args(); print({"games": len(process_game_logs(a.input_dir, a.output_dir, a.naming))})

if __name__ == "__main__": main()
