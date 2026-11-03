"""Small Gemini REST adapter. No credentials in URLs, source, or error messages."""
from __future__ import annotations
import os
import time
from types import SimpleNamespace
from urllib.parse import quote
import requests


def api_key():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key: raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
    return key


def generate_text(model, prompt, *, key=None, max_tokens=8192, temperature=0.0,
                  top_p=1.0, json_output=False, system=None, retries=3,
                  generation_config=None, return_metadata=False, allow_truncated=False):
    key = key or api_key()
    if retries < 1 or max_tokens < 1: raise ValueError("Positive retry and token counts are required")
    config = {"temperature": temperature, "topP": top_p, "maxOutputTokens": max_tokens}
    if json_output: config["responseMimeType"] = "application/json"
    if generation_config: config.update(generation_config)
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": config}
    if system: body["systemInstruction"] = {"parts": [{"text": system}]}
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model, safe='-._')}:generateContent"
    for attempt in range(retries):
        try:
            r = requests.post(endpoint, headers={"x-goog-api-key": key}, json=body, timeout=(15, 180))
            if r.status_code in {429, 500, 502, 503, 504}:
                if attempt + 1 < retries: time.sleep(min(2 ** attempt, 16)); continue
            if not r.ok: raise RuntimeError(f"Gemini request failed (HTTP {r.status_code}); check model access and account settings")
            obj = r.json(); candidates = obj.get("candidates", [])
            if not candidates: raise RuntimeError("Gemini returned no candidate (possibly blocked)")
            c = candidates[0]
            finish = c.get("finishReason", "")
            if finish in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "OTHER"} or (finish == "MAX_TOKENS" and not allow_truncated):
                raise RuntimeError(f"Gemini response is not complete: {finish}")
            text = "".join(p.get("text", "") for p in c.get("content", {}).get("parts", []) if not p.get("thought"))
            if not text.strip(): raise RuntimeError("Gemini returned no answer text")
            if return_metadata:
                usage = obj.get("usageMetadata", {})
                return {"generated_text": text.strip(), "finish_reason": finish,
                        "prompt_tokens": usage.get("promptTokenCount"),
                        "completion_tokens": usage.get("candidatesTokenCount"),
                        "thought_tokens": usage.get("thoughtsTokenCount")}
            return text.strip()
        except requests.RequestException:
            if attempt + 1 == retries: raise RuntimeError("Gemini network request failed") from None
            time.sleep(min(2 ** attempt, 16))
    raise RuntimeError("Gemini request did not complete")


class Model:
    """Per-worker credentials; compatible with the construction call interface."""
    def __init__(self, model_name, key=None, max_tokens=8192):
        self.model_name, self.key, self.max_tokens = model_name, key, max_tokens
    def generate_content(self, prompt, **kwargs):
        text = generate_text(self.model_name, prompt, key=self.key, max_tokens=self.max_tokens)
        return SimpleNamespace(text=text, parts=[SimpleNamespace(text=text)])
