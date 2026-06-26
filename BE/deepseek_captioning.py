"""DeepSeek TEXT-only reasoning reranker for /identify_and_search (Pass 2).

Matches the Sonnet 7-field caption TEXT against the candidate reference descriptions
TEXT using DeepSeek's reasoning model (no image). In benchmarking this beat the Haiku
vision rerank on top-1 accuracy (0.633 vs 0.478) at a fraction of the cost.

The prompt, candidate formatting, parsing, and output shaping are shared with the
Haiku text fallback via text_rerank, so both rerankers behave identically.
"""
import os

import requests

from text_rerank import (
    TEXT_RERANK_SYSTEM,
    build_user_prompt,
    parse_json_object,
    shape_results,
)

DS_URL = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com/chat/completions")
DS_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DS_TIMEOUT = int(os.getenv("DEEPSEEK_TIMEOUT", "180"))
# Reasoning model: reasoning_content is billed in completion_tokens, so give
# generous headroom or content comes back empty (finish_reason=length). Hard images
# can spend >8k tokens reasoning; 16384 lets them finish and emit the JSON.
DS_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", "16384"))


def _get_api_key():
    return os.getenv("DEEPSEEK_API_KEY")


def _request_deepseek(messages):
    """POST to DeepSeek, retrying without JSON mode if the model rejects it (HTTP 400).

    Returns the message content string. Raises RuntimeError on empty content
    (reasoning consumed the whole budget) so the caller can fall back.
    """
    key = _get_api_key()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")
    base = {"model": DS_MODEL, "messages": messages, "max_tokens": DS_MAX_TOKENS, "stream": False}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last_exc = None
    for body in ({**base, "temperature": 0, "response_format": {"type": "json_object"}}, base):
        r = requests.post(DS_URL, headers=headers, json=body, timeout=DS_TIMEOUT)
        if r.status_code == 400:
            last_exc = RuntimeError(f"DeepSeek HTTP 400: {r.text[:200]}")
            continue
        r.raise_for_status()
        content = (r.json().get("choices") or [{}])[0].get("message", {}).get("content") or ""
        if not content.strip():
            raise RuntimeError("DeepSeek returned empty content (reasoning consumed the token budget)")
        return content
    raise last_exc or RuntimeError(f"DeepSeek request failed for model {DS_MODEL}")


def chat_deepseek(messages, max_tokens=2000, temperature=0):
    """Plain chat completion via DeepSeek for /generation (the fish chatbot).

    Unlike _request_deepseek this does NOT force JSON mode — it returns free-form
    markdown prose. `messages` is an OpenAI-style list (system/user/assistant);
    DeepSeek is OpenAI-compatible so the system role is passed through as-is.
    """
    key = _get_api_key()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": DS_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    r = requests.post(DS_URL, headers=headers, json=body, timeout=DS_TIMEOUT)
    r.raise_for_status()
    content = (r.json().get("choices") or [{}])[0].get("message", {}).get("content") or ""
    if not content.strip():
        raise RuntimeError("DeepSeek returned empty content")
    return content.strip()


def chat_deepseek(messages, max_tokens=1024, temperature=0.0, model=None):
    """Free-form chat completion via DeepSeek (used by generation.py for RAG answers).

    `messages` is an OpenAI-style list ({"role": "system"|"user"|"assistant", "content": ...}).
    Unlike the reranker this does NOT force JSON mode. Returns the assistant text; raises
    RuntimeError on empty content so the caller can fall back to another provider.
    """
    key = _get_api_key()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")
    body = {"model": model or DS_MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "stream": False}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    r = requests.post(DS_URL, headers=headers, json=body, timeout=DS_TIMEOUT)
    r.raise_for_status()
    content = (r.json().get("choices") or [{}])[0].get("message", {}).get("content") or ""
    if not content.strip():
        raise RuntimeError("DeepSeek returned empty content")
    return content


def rerank_candidates_deepseek(caption, candidates):
    """Re-rank an ES shortlist by matching the caption TEXT against candidate descriptions.

    `caption` is the 7-field description string from the vision caption step.
    `candidates` is a list of dicts (from function.return_top_n_fish). Returns the shared
    text-rerank shape: {"image_contains_fish": None, "rejection_reason": None,
    "observed_features": <dict|None>, "results": [...]} — same as the Haiku text fallback.
    """
    content = _request_deepseek([
        {"role": "system", "content": TEXT_RERANK_SYSTEM},
        {"role": "user", "content": build_user_prompt(caption, candidates)},
    ])
    return shape_results(caption, candidates, parse_json_object(content))
