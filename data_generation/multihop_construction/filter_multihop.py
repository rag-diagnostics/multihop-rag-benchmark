import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _extract_entries(obj: Any) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Returns (entries_list, container_key).
    - If top-level is a list, container_key=None.
    - If top-level is a dict that contains a list under a common key, container_key=<that key>.
    """
    if isinstance(obj, list):
        return obj, None

    if isinstance(obj, dict):
        for key in ("data", "items", "records", "examples"):
            if key in obj and isinstance(obj[key], list):
                return obj[key], key

    raise ValueError("Unsupported JSON shape: expected a list, or a dict containing a list under data/items/records/examples.")


def _entry_has_hops_error(entry: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Checks whether entry['hops'] contains ANY node with an 'error' field.
    Returns (has_error, error_node_ids)
    """
    hops = entry.get("hops", {})
    if not isinstance(hops, dict):
        return False, []

    bad_nodes = []
    for node_id, node in hops.items():
        if isinstance(node, dict) and "error" in node:
            bad_nodes.append(str(node_id))

    return (len(bad_nodes) > 0), bad_nodes


def filter_json_remove_entries_with_hops_error(input_path: str, output_path: str) -> None:
    in_path = Path(input_path)
    out_path = Path(output_path)

    with in_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    entries, container_key = _extract_entries(raw)

    before_count = len(entries)
    before_questions = [e.get("original_question") for e in entries]

    kept: List[Dict[str, Any]] = []
    removed_info: List[Tuple[int, str, List[str]]] = []  # (index, original_question, bad_node_ids)

    for i, e in enumerate(entries):
        has_err, bad_nodes = _entry_has_hops_error(e)
        if has_err:
            removed_info.append((i, str(e.get("original_question", "")), bad_nodes))
        else:
            kept.append(e)

    after_count = len(kept)
    after_questions = [e.get("original_question") for e in kept]

    # Rebuild output JSON in same "shape" as input
    if container_key is None:
        out_obj: Any = kept
    else:
        out_obj = dict(raw)
        out_obj[container_key] = kept

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    # Print "previous and new saved json file entities"
    print("=== SUMMARY ===")
    print(f"Input file:  {in_path.resolve()}")
    print(f"Output file: {out_path.resolve()}")
    print(f"Entries BEFORE: {before_count}")
    print(f"Entries AFTER : {after_count}")
    print(f"Removed       : {before_count - after_count}")
    print()

    print("=== ORIGINAL (BEFORE) original_question entries ===")
    for idx, q in enumerate(before_questions):
        print(f"[{idx}] {q}")

    print("\n=== REMOVED (had hops error) ===")
    if not removed_info:
        print("(none)")
    else:
        for idx, q, bad_nodes in removed_info:
            print(f"[{idx}] bad hop nodes={bad_nodes} | original_question={q}")

    print("\n=== NEW (AFTER) original_question entries ===")
    for idx, q in enumerate(after_questions):
        print(f"[{idx}] {q}")


if __name__ == "__main__":
    # Change these two paths:
    INPUT_JSON = "000_new_multihop_dataset_final.json"
    OUTPUT_JSON = "filtered_000_new_multihop_dataset_final.json"

    filter_json_remove_entries_with_hops_error(INPUT_JSON, OUTPUT_JSON)

