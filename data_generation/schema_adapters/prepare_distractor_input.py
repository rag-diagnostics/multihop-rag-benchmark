"""Use transformed narratives explicitly; retain stable IDs; avoid lossy workbook writes."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag_utils.tables import read_records, write_records, ensure_serials


def prepare(src, dest, text_column="Summary", document_column="Document"):
    if Path(src).resolve() == Path(dest).resolve(): raise ValueError("Input and output must differ")
    rows = ensure_serials(read_records(src))
    for row in rows:
        for name in (text_column, "question", "answer"):
            if not isinstance(row.get(name), str) or not row[name].strip(): raise ValueError("Missing nonempty " + name)
        if row.get(document_column) and row[document_column] != row[text_column]: raise ValueError("Existing Document conflicts with transformed Summary")
        row[document_column] = row[text_column]
    write_records(dest, rows); return len(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--text-column", default="Summary"); p.add_argument("--document-column", default="Document")
    a = p.parse_args(); print(prepare(a.input, a.output, a.text_column, a.document_column))

if __name__ == "__main__": main()
