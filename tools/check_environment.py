"""Report dependencies and credential presence without reading or displaying key values."""
import importlib.util
import os
import json
import sys

def main():
    packages = ["requests", "gdown", "openai", "pandas", "openpyxl", "spacy", "nltk", "torch", "transformers", "accelerate", "dotenv", "tqdm"]
    report = {"python": sys.version.split()[0], "packages": {p: importlib.util.find_spec(p) is not None for p in packages},
              "credential_variables_present": {k: bool(os.getenv(k)) for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "DEEPINFRA_TOKEN", "OPENROUTER_API_KEY")}}
    print(json.dumps(report, indent=2))
if __name__ == "__main__": main()
