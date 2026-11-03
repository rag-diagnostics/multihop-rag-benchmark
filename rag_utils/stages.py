"""Per-row construction checkpoints with input and protocol fingerprints."""
from pathlib import Path
from .io import fingerprint, atomic_json, read_json, output_lock
from .tables import ensure_serials, write_records


def run_rows(rows, output, protocol, process, required_fields):
    rows = ensure_serials(rows)
    out = Path(output); cache = Path(str(out) + ".checkpoint.json")
    if out.resolve() == cache.resolve(): raise ValueError("Invalid checkpoint path")
    identity = fingerprint({"protocol": protocol, "inputs": rows})
    with output_lock(out):
        if out.exists() and not cache.exists():
            raise ValueError("Existing output has no matching checkpoint; choose a new output path")
        saved = read_json(cache) if cache.exists() else {"identity": identity, "protocol": protocol, "records": {}}
        if saved.get("identity") != identity: raise ValueError("Construction inputs/settings changed; use a new output path")
        failures = []
        for row in rows:
            serial = row["Serial"]
            if saved["records"].get(serial, {}).get("status") == "ok": continue
            try:
                result = dict(row); result.update(process(dict(row)))
                if result["Serial"] != serial: raise ValueError("A stage changed the source identifier")
                for field in required_fields:
                    if not isinstance(result.get(field), str) or not result[field].strip(): raise ValueError("Missing output: " + field)
                saved["records"][serial] = {"status": "ok", "row": result}
            except Exception as exc:
                saved["records"][serial] = {"status": "error", "error_type": type(exc).__name__}
                failures.append(serial)
            atomic_json(cache, saved)
        completed = [saved["records"][r["Serial"]]["row"] for r in rows if saved["records"].get(r["Serial"], {}).get("status") == "ok"]
        atomic_json(str(out) + ".status.json", {"requested": len(rows), "completed": len(completed), "failed_serials": failures, "protocol": protocol})
        if len(completed) != len(rows):
            raise RuntimeError(f"Stage incomplete: {len(completed)}/{len(rows)} records. Check {out.name}.status.json; rerun to retry failures.")
        write_records(out, completed)
    return completed
