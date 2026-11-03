"""Select source transcripts reproducibly; never claim newly sampled IDs are historical IDs."""
import argparse
import random
from pathlib import Path
from data_generation.narrative_generation.generate_narratives import collect_messages
from rag_utils.io import atomic_json, file_sha256


def select(llm_dir, qanta_dir, output, llm_count=300, qanta_count=300, seed=42, pattern="*.txt"):
    if min(llm_count, qanta_count) < 1: raise ValueError("Both source counts must be positive")
    rng = random.Random(seed); result = []
    for label, directory, count in (("llm", llm_dir, llm_count), ("qanta", qanta_dir, qanta_count)):
        rows = collect_messages(directory, pattern, label)
        if len(rows) < count: raise ValueError(f"Need {count} {label} samples; found {len(rows)}")
        selected = sorted(rng.sample(rows, count), key=lambda r: r["source_filename"])
        for i, row in enumerate(selected, 1):
            row["Serial"] = f"{label}_{i:06d}"
            result.append(row)
    atomic_json(output, result)
    atomic_json(str(output) + ".selection.json", {"seed": seed, "llm_count": llm_count, "qanta_count": qanta_count,
        "selected_sources": [{k: r[k] for k in ("Serial", "source", "source_filename", "source_sha256")} for r in result]})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--llm-dir", required=True); p.add_argument("--qanta-dir", required=True)
    p.add_argument("--output", required=True); p.add_argument("--llm-count", type=int, default=300)
    p.add_argument("--qanta-count", type=int, default=300); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pattern", default="*.txt")
    a = p.parse_args(); select(a.llm_dir, a.qanta_dir, a.output, a.llm_count, a.qanta_count, a.seed, a.pattern)

if __name__ == "__main__": main()
