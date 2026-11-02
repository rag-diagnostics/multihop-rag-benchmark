#!/usr/bin/env python3
"""Test local vLLM OpenAI-compatible /v1/models endpoints, without API keys."""
import argparse
import json
from urllib.error import URLError
from urllib.request import urlopen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urls", nargs="+", default=["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"])
    args = parser.parse_args()
    for endpoint in args.urls:
        if not endpoint.startswith(("http://127.0.0.1:", "http://localhost:")):
            parser.error("Only local endpoints are supported")
        url = endpoint.rstrip("/") + "/models"
        try:
            with urlopen(url, timeout=4) as response:
                body = json.load(response)
            ids = [item.get("id", "?") for item in body.get("data", [])]
            print(f"OK {url}: {', '.join(ids) if ids else 'no models reported'}")
        except (URLError, OSError, ValueError) as exc:
            print(f"FAILED {url}: {type(exc).__name__}: {exc}")
            raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
