"""Model adapters. Explicit settings; no silent model substitution or input truncation."""
from __future__ import annotations
import os
import threading
import time
from functools import lru_cache
from typing import Any, Dict

_THREAD_LOCAL = threading.local()


def validate_models(config):
    if not isinstance(config, dict) or not isinstance(config.get("models"), dict) or not config["models"]:
        raise ValueError("Model config must contain a nonempty 'models' object")
    def reject_secrets(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() in {"api_key", "token", "authorization", "password", "secret", "headers"}:
                    raise ValueError("Do not put credentials or headers in model config; use api_key_env")
                reject_secrets(v)
        elif isinstance(obj, list):
            for v in obj: reject_secrets(v)
    reject_secrets(config)
    for key, cfg in config["models"].items():
        if not isinstance(key, str) or not key.strip() or "|" in key: raise ValueError("Model keys must be nonempty and must not contain |")
        if not isinstance(cfg, dict) or cfg.get("provider") not in {"openai_compatible", "google_genai", "transformers"}:
            raise ValueError(f"Unsupported provider for {key}")
        if not isinstance(cfg.get("model_id"), str) or not cfg["model_id"].strip(): raise ValueError("Missing model_id")
        if cfg["provider"] != "transformers" and not cfg.get("api_key_env"): raise ValueError("Missing api_key_env")
        if cfg["provider"] == "openai_compatible" and not str(cfg.get("base_url", "")).startswith(("https://", "http://localhost", "http://127.0.0.1")):
            raise ValueError("API base_url must be HTTPS, or a local server")
        budget = cfg.get("max_tokens", 1)
        if type(budget) is not int or budget < 1: raise ValueError("max_tokens must be a positive integer")
    return config["models"]


def _api_key(cfg):
    name = cfg["api_key_env"]; value = os.getenv(name, "")
    if not value: raise RuntimeError(f"Missing environment variable: {name}")
    return value


def preflight(cfg):
    if cfg["provider"] != "transformers": _api_key(cfg)
    if cfg["provider"] == "openai_compatible":
        import openai
    elif cfg["provider"] == "transformers":
        import torch, transformers


def _client_cache():
    if not hasattr(_THREAD_LOCAL, "clients"): _THREAD_LOCAL.clients = {}
    return _THREAD_LOCAL.clients


@lru_cache(maxsize=1)
def _hf_model(model_id, revision=None):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, device_map="auto", torch_dtype="auto", trust_remote_code=False)
    model.eval(); return tok, model


def release_local_models():
    _hf_model.cache_clear()
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except ImportError: pass


def generate(cfg: Dict[str, Any], prompt: str, max_tokens: int) -> Dict[str, Any]:
    started = time.perf_counter(); provider = cfg["provider"]
    if cfg.get("max_input_characters") and len(prompt) > cfg["max_input_characters"]:
        raise ValueError("Prompt exceeds configured character guard; no text was truncated")
    temp = cfg.get("temperature", 0.0); top_p = cfg.get("top_p", 1.0)
    if provider == "openai_compatible":
        from openai import OpenAI
        cache = _client_cache(); ck = (provider, cfg["base_url"], cfg["api_key_env"])
        if ck not in cache:
            cache[ck] = OpenAI(api_key=_api_key(cfg), base_url=cfg["base_url"], timeout=cfg.get("timeout", 180), max_retries=0)
        kw = {"model": cfg["model_id"], "messages": [{"role": "user", "content": prompt}]}
        token_parameter = cfg.get("token_parameter", "max_tokens")
        if token_parameter not in {"max_tokens", "max_completion_tokens"}: raise ValueError("Invalid token_parameter")
        kw[token_parameter] = max_tokens
        if temp is not None: kw["temperature"] = temp
        if top_p is not None: kw["top_p"] = top_p
        if cfg.get("extra_body"): kw["extra_body"] = cfg["extra_body"]
        resp = cache[ck].chat.completions.create(**kw)
        if not getattr(resp, "choices", None): raise RuntimeError("No completion choices")
        c = resp.choices[0]; text = c.message.content
        if text is None: raise RuntimeError("No answer text returned")
        usage = getattr(resp, "usage", None)
        result = {"generated_text": text.strip(), "finish_reason": getattr(c, "finish_reason", None),
                  "prompt_tokens": getattr(usage, "prompt_tokens", None), "completion_tokens": getattr(usage, "completion_tokens", None)}
    elif provider == "google_genai":
        from rag_utils.gemini import generate_text
        result = generate_text(cfg["model_id"], prompt, key=_api_key(cfg), max_tokens=max_tokens,
            temperature=temp, top_p=top_p, retries=1, generation_config=cfg.get("generation_config"),
            return_metadata=True, allow_truncated=True)
    elif provider == "transformers":
        import torch
        tok, model = _hf_model(cfg["model_id"], cfg.get("revision"))
        if getattr(tok, "chat_template", None):
            text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            batch = tok(text, return_tensors="pt", add_special_tokens=False)
        else: batch = tok(prompt, return_tensors="pt")
        n = int(batch["input_ids"].shape[-1])
        max_context = cfg.get("max_context_tokens") or getattr(model.config, "max_position_embeddings", None)
        if max_context and n + max_tokens > max_context:
            raise ValueError("Prompt plus output budget exceeds model context window; no truncation performed")
        batch = batch.to(model.get_input_embeddings().weight.device)
        with torch.inference_mode():
            output = model.generate(**batch, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        ids = output[0][n:]
        result = {"generated_text": tok.decode(ids, skip_special_tokens=True).strip(),
                  "finish_reason": "length" if len(ids) >= max_tokens else "stop", "prompt_tokens": n, "completion_tokens": int(len(ids))}
    else: raise ValueError("Unsupported provider")
    result["time_seconds"] = round(time.perf_counter() - started, 4)
    return result
