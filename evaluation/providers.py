"""Stateless model invocation with lazy SDK imports; credentials from environment only."""
from __future__ import annotations
import os
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

_THREAD_LOCAL = threading.local()


def validate_models(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(config, dict) or not isinstance(config.get("models"), dict):
        raise ValueError("Model config must contain a 'models' object")
    models = config["models"]
    if not models:
        raise ValueError("At least one model required")
    for model_key, cfg in models.items():
        if not isinstance(cfg, dict) or cfg.get("provider") not in ("openai_compatible", "google_genai", "transformers"):
            raise ValueError(f"Unsupported provider for {model_key}")
        if not isinstance(cfg.get("model_id"), str) or not cfg["model_id"].strip():
            raise ValueError(f"Missing model_id for {model_key}")
        if cfg["provider"] in ("openai_compatible", "google_genai"):
            if not isinstance(cfg.get("api_key_env"), str) or not cfg["api_key_env"].strip():
                raise ValueError(f"Missing api_key_env for {model_key}")
        if cfg["provider"] == "openai_compatible" and not cfg.get("base_url"):
            raise ValueError(f"Missing base_url for {model_key}")
    return models


def _client_cache() -> Dict[tuple, Any]:
    if not hasattr(_THREAD_LOCAL, "clients"):
        _THREAD_LOCAL.clients = {}
    return _THREAD_LOCAL.clients


def _api_key(cfg: Dict[str, Any]) -> str:
    name = cfg["api_key_env"]
    value = os.getenv(name, "")
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


@lru_cache(maxsize=3)
def _hf_model(model_id: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype="auto", trust_remote_code=False
    )
    model.eval()
    return tok, model


def generate(cfg: Dict[str, Any], prompt: str, max_tokens: int) -> Dict[str, Any]:
    """Greedy/top-p-1 generation. Provider/model support for all parameters is required."""
    started = time.perf_counter()
    provider = cfg["provider"]
    if provider == "openai_compatible":
        from openai import OpenAI
        cache = _client_cache()
        cache_key = (provider, cfg["base_url"], cfg["api_key_env"])
        if cache_key not in cache:
            cache[cache_key] = OpenAI(api_key=_api_key(cfg), base_url=cfg["base_url"], timeout=120.0)
        resp = cache[cache_key].chat.completions.create(
            model=cfg["model_id"], messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.0, top_p=1.0
        )
        if not getattr(resp, "choices", None):
            raise RuntimeError("Model returned no completion choices")
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return {"generated_text": (choice.message.content or "").strip(),
                "finish_reason": getattr(choice, "finish_reason", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "time_seconds": round(time.perf_counter()-started, 4)}
    if provider == "google_genai":
        from google import genai
        from google.genai import types
        cache = _client_cache()
        cache_key = (provider, cfg["api_key_env"])
        if cache_key not in cache:
            cache[cache_key] = genai.Client(api_key=_api_key(cfg))
        resp = cache[cache_key].models.generate_content(
            model=cfg["model_id"], contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, top_p=1.0,
                                               max_output_tokens=max_tokens)
        )
        usage = getattr(resp, "usage_metadata", None)
        return {"generated_text": (getattr(resp,"text","") or "").strip(),
                "finish_reason": None,
                "prompt_tokens": getattr(usage, "prompt_token_count", None),
                "completion_tokens": getattr(usage, "candidates_token_count", None),
                "time_seconds": round(time.perf_counter()-started, 4)}
    if provider == "transformers":
        import torch
        tok, model = _hf_model(cfg["model_id"])
        if getattr(tok, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            encoded = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                              return_tensors="pt")
        else:
            encoded = tok(prompt, return_tensors="pt")["input_ids"]
        encoded = encoded.to(model.device)
        with torch.inference_mode():
            output = model.generate(encoded, max_new_tokens=max_tokens, do_sample=False,
                                    pad_token_id=tok.eos_token_id)
        generated = tok.decode(output[0][encoded.shape[-1]:], skip_special_tokens=True)
        return {"generated_text": generated.strip(), "finish_reason": None,
                "prompt_tokens": int(encoded.shape[-1]),
                "completion_tokens": int(output.shape[-1] - encoded.shape[-1]),
                "time_seconds": round(time.perf_counter()-started, 4)}
    raise ValueError(f"Unsupported provider: {provider}")
