"""JSON is the lossless canonical format; Excel export fails before truncation."""
from pathlib import Path
import re
from .io import read_json, atomic_json


def ensure_serials(rows):
    seen = set(); result = []
    for i, row in enumerate(rows, 1):
        r = dict(row)
        serial = r.get("Serial")
        if serial is None or str(serial).strip() == "":
            serial = f"sample_{i:06d}"
        serial = str(serial)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", serial) or serial in {".", ".."}:
            raise ValueError("Serial must be a safe, non-empty identifier")
        if serial in seen: raise ValueError("Duplicate Serial; assign unique source IDs before processing")
        seen.add(serial); r["Serial"] = serial; result.append(r)
    return result


def read_records(path):
    p = Path(path)
    if p.suffix.lower() in {".json", ".jsonl"}:
        if p.suffix.lower() == ".jsonl":
            from .io import iter_jsonl
            rows = list(iter_jsonl(p))
        else: rows = read_json(p)
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            raise ValueError("Expected a list of records")
        return rows
    if p.suffix.lower() == ".xlsx":
        from openpyxl import load_workbook
        wb = load_workbook(p, read_only=True, data_only=False)
        try:
            ws = wb.active; ws.reset_dimensions(); it = ws.iter_rows(values_only=True)
            header = next(it, ())
            names = [str(c) if c is not None else "" for c in header]
            if not names or "" in names or len(set(names)) != len(names):
                raise ValueError("Workbook requires unique nonempty column names")
            return [dict(zip(names, values)) for values in it if any(v is not None for v in values)]
        finally: wb.close()
    raise ValueError("Use .json, .jsonl, or .xlsx")


def write_records(path, rows):
    p = Path(path)
    if p.suffix.lower() == ".json": atomic_json(p, rows); return
    if p.suffix.lower() != ".xlsx": raise ValueError("Output must be .json or .xlsx")
    # Check BEFORE openpyxl has the chance to silently truncate text.
    for i, row in enumerate(rows):
        for key, value in row.items():
            if isinstance(value, str) and len(value) > 32767:
                raise ValueError(f"Row {i}, {key}: exceeds Excel's 32767-character cell limit; use .json output")
            if isinstance(value, (dict, list)):
                raise ValueError(f"Row {i}, {key}: nested data requires .json output")
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    wb = Workbook(); ws = wb.active; ws.title = "Records"
    names = list(dict.fromkeys(k for r in rows for k in r))
    ws.append(names); ws.freeze_panes = "A2"
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF"); cell.fill = PatternFill("solid", fgColor="254B5D")
    for row in rows:
        ws.append([row.get(k) for k in names])
        for cell in ws[ws.max_row]:
            if isinstance(cell.value, str): cell.data_type = "s"  # no formula injection
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.xlsx")
    try: wb.save(tmp); tmp.replace(p)
    finally: wb.close(); tmp.unlink(missing_ok=True)
