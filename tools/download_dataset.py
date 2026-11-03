"""Download the published Drive JSON atomically; verify the full record count."""
from __future__ import annotations
import argparse
from pathlib import Path
from rag_utils.io import iter_json_array, file_sha256, atomic_json

FILE_ID = "1GmHj2K5SCaV0-a8xFjaj9UprHv6hbmXa"
URL = f"https://drive.google.com/file/d/{FILE_ID}/view?usp=sharing"


def verify(path, expected=528, sha256=None):
    count = sum(1 for _ in iter_json_array(path))
    if expected is not None and count != expected: raise ValueError(f"Expected {expected} records, found {count}")
    digest = file_sha256(path)
    if sha256 and digest.lower() != sha256.lower(): raise ValueError("SHA-256 mismatch; existing dataset was not replaced")
    return {"records": count, "sha256": digest, "bytes": Path(path).stat().st_size}


def download(output, expected=528, sha256=None, force=False, fetch=None):
    p = Path(output); p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not force:
        report = verify(p, expected, sha256); report["downloaded"] = False
        return report
    tmp = p.with_name(p.name + ".download")
    try:
        if fetch is None:
            try: import gdown
            except ImportError: raise RuntimeError("Install gdown, or download the Drive file in a browser and use --verify-only") from None
            fetch = lambda dest: gdown.download(id=FILE_ID, output=str(dest), quiet=False, use_cookies=False)
        result = fetch(tmp)
        if not result or not tmp.is_file(): raise RuntimeError("Drive download failed. Check public sharing or download in your browser.")
        report = verify(tmp, expected, sha256)
        tmp.replace(p); report["downloaded"] = True
        atomic_json(str(p) + ".download.json", {**report, "source": URL})
        return report
    finally: tmp.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("data/benchmark_528.json"))
    p.add_argument("--expected-chains", type=int, default=528)
    p.add_argument("--sha256", help="Publisher checksum when available; no checksum is invented")
    p.add_argument("--force", action="store_true")
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()
    try:
        report = verify(a.output, a.expected_chains, a.sha256) if a.verify_only else download(a.output, a.expected_chains, a.sha256, a.force)
        print(report)
    except (ValueError, RuntimeError, OSError) as e:
        p.exit(2, f"Dataset access failed: {e}\n")

if __name__ == "__main__": main()
