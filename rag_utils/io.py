"""Lossless storage, strict streaming JSON input, and resumable experiment identities."""
from __future__ import annotations
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def _no_duplicates(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"Duplicate JSON key: {key}")
        out[key] = value
    return out


def _bad_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


DECODER = json.JSONDecoder(object_pairs_hook=_no_duplicates, parse_constant=_bad_constant)


def fingerprint(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return DECODER.decode(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, obj, *, indent=2):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        tmp.replace(p)
    finally:
        tmp.unlink(missing_ok=True)


def append_jsonl(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush(); os.fsync(handle.fileno())


def iter_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            if not line.strip(): continue
            try: obj = DECODER.decode(line)
            except (ValueError, json.JSONDecodeError) as e:
                raise ValueError(f"Invalid JSONL at line {i}; restore a truncated final line before resuming") from e
            if not isinstance(obj, dict): raise ValueError(f"Expected object at JSONL line {i}")
            yield obj


def iter_json_array(path, chunk_size=65536) -> Iterator[dict]:
    """Read one record at a time; reject truncated arrays, duplicate keys and trailing data.

    The entire file is validated by callers before a paid inference run. No recovery
    of partially pasted JSON is performed in production code.
    """
    if chunk_size < 1: raise ValueError("chunk_size must be positive")
    with Path(path).open(encoding="utf-8-sig") as f:
        buf = ""; eof = False
        def refill():
            nonlocal buf, eof
            text = f.read(chunk_size)
            if text: buf += text
            else: eof = True
        def skip():
            nonlocal buf
            while True:
                buf = buf.lstrip()
                if buf or eof: return
                refill()
        skip()
        if not buf.startswith("["): raise ValueError("Dataset must be a JSON array, not HTML or a Drive preview page")
        buf = buf[1:]; first = True
        while True:
            skip()
            if buf.startswith("]"):
                buf = buf[1:]; break
            if not first:
                if not buf.startswith(","): raise ValueError("Expected comma or closing bracket in dataset")
                buf = buf[1:]; skip()
                if buf.startswith("]"): raise ValueError("Trailing comma in dataset")
            while True:
                try:
                    row, end = DECODER.raw_decode(buf); break
                except json.JSONDecodeError as e:
                    if eof: raise ValueError("Incomplete or invalid dataset JSON; download the full file") from e
                    refill()
            if not isinstance(row, dict): raise ValueError("Every dataset record must be an object")
            buf = buf[end:]; first = False
            yield row
        skip()
        if buf: raise ValueError("Trailing non-whitespace content after dataset array")


@contextmanager
def output_lock(path):
    """Prevent concurrent writers; a stale lock requires explicit manual removal."""
    p = Path(str(path) + ".lock"); p.parent.mkdir(parents=True, exist_ok=True)
    try: fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(f"Output is locked: {p}. Remove a stale lock only after checking no run is active.") from None
    try:
        os.write(fd, str(os.getpid()).encode("ascii")); os.close(fd)
        yield
    finally: p.unlink(missing_ok=True)


def require_new_manifest(path, identity):
    """Refuse resume after dataset, settings, sample, code, or model configuration changes."""
    p = Path(path)
    if p.exists():
        prior = read_json(p)
        if prior.get("identity_sha256") != fingerprint(identity):
            raise ValueError("Run configuration changed. Use a new output directory; existing results are untouched.")
        return prior
    obj = {"identity_sha256": fingerprint(identity), "identity": identity}
    atomic_json(p, obj)
    return obj

class JsonlIndex:
    """Index latest request offsets without retaining long contexts in memory."""
    def __init__(self, path):
        self.path = Path(path); self.entries = {}
        if not self.path.exists(): return
        with self.path.open("rb") as f:
            line_no = 0
            while True:
                offset = f.tell(); raw = f.readline()
                if not raw: break
                line_no += 1
                if not raw.strip(): continue
                try: row = DECODER.decode(raw.decode("utf-8-sig"))
                except ValueError as e: raise ValueError(f"Invalid JSONL at {self.path}:{line_no}") from e
                if not isinstance(row, dict) or not isinstance(row.get("request_id"), str):
                    raise ValueError(f"Missing request_id at line {line_no}")
                self.entries[row["request_id"]] = {"offset": offset, **{k: row.get(k) for k in (
                    "status", "request_fingerprint", "judgment_fingerprint", "judgment_status")}}
    def get(self, key):
        if key not in self.entries: return None
        with self.path.open("rb") as f:
            f.seek(self.entries[key]["offset"])
            return DECODER.decode(f.readline().decode("utf-8-sig"))
    def rows(self):
        if not self.entries: return
        with self.path.open("rb") as f:
            for meta in sorted(self.entries.values(), key=lambda v: v["offset"]):
                f.seek(meta["offset"])
                yield DECODER.decode(f.readline().decode("utf-8-sig"))
