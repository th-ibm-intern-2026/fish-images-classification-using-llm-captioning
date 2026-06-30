"""DeepSeek TEXT-only reasoning reranker for /identify_and_search (Pass 2).

Matches the Sonnet 7-field caption TEXT against the candidate reference descriptions
TEXT using DeepSeek's reasoning model (no image). DeepSeek is the only reranker now
(the Haiku text fallback was removed), so the rerank prompt, candidate formatting,
JSON parsing, and result shaping live here rather than in a shared module.
"""
import json
import os

import requests

from fish_constants import PHYSICAL_FEATURE_FIELDS

DS_URL = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com/chat/completions")
DS_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DS_TIMEOUT = int(os.getenv("DEEPSEEK_TIMEOUT", "180"))
DS_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", "16384"))

# Rerank prompt: matches the OBSERVED 7-field caption against the candidate shortlist.
TEXT_RERANK_SYSTEM = (
    "You are an expert ichthyologist. You are given an OBSERVED description of a fish read "
    "from a photo using a fixed 7-field template (body_shape; primary_colors; secondary_colors; "
    "markings; dorsal_fin; caudal_fin; unique_features), and a numbered SHORTLIST of candidate "
    "species, each with its own 7-field reference description. Decide which candidate the observed "
    "fish actually is. Compare field-by-field; weight agreement in unique_features and markings most "
    "heavily; a single matching color must NOT outweigh a clear mismatch in body_shape or markings. "
    "Choose ONLY from the candidates. Do your comparison in your reasoning, then output ONLY "
    "compact valid JSON (no markdown, no per-item prose):\n"
    '{"results": [{"scientific_name": <copied from candidate>, "match_score": <float 0-1>}]} '
    "ordered best-match first, exactly one object per candidate."
)


def _get_api_key():
    return os.getenv("DEEPSEEK_API_KEY")


def parse_json_object(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in rerank output: {text[:200]}")
    return json.loads(text[start : end + 1])


def candidate_text(candidates):
    return "\n".join(
        f"{i}. {c.get('fish_name')} (scientific_name: {c.get('scientific_name')})\n"
        f"   description: {c.get('physical_description') or 'not available'}"
        for i, c in enumerate(candidates, 1)
    )


def build_user_prompt(caption, candidates):
    return f"OBSERVED (from a photo):\n{caption}\n\nCANDIDATES:\n{candidate_text(candidates)}"


def parse_caption_features(caption):
    """Turn the 7-field caption string ('label: value; ...') into a dict (best-effort)."""
    features = {}
    for part in (caption or "").split(";"):
        if ":" not in part:
            continue
        label, _, value = part.partition(":")
        key = label.strip().lower().replace(" ", "_")
        if key in PHYSICAL_FEATURE_FIELDS:
            features[key] = value.strip()
    return features or None


def shape_results(caption, candidates, parsed):
    """Join the reranker's (scientific_name, match_score) back onto the full candidate
    dicts and return the shape /identify_and_search expects. image_contains_fish stays
    None (the text reranker never sees the image; the gate owns that verdict).
    """
    by_sci = {(c.get("scientific_name") or "").strip().lower(): c for c in candidates}
    results, seen = [], set()
    for item in parsed.get("results") or []:
        sci = (item.get("scientific_name") or "").strip()
        cand = by_sci.get(sci.lower())
        if cand is None:
            continue
        seen.add(sci.lower())
        results.append({**cand, "rerank_score": item.get("match_score"), "rerank_reason": ""})
    # Append any candidate the reranker omitted, preserving ES order, so the list is complete.
    for c in candidates:
        if (c.get("scientific_name") or "").strip().lower() not in seen:
            results.append({**c, "rerank_score": None, "rerank_reason": ""})
    return {
        "image_contains_fish": None,
        "rejection_reason": None,
        "observed_features": parse_caption_features(caption),
        "results": results,
    }


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
    `candidates` is a list of dicts (from function.return_top_n_fish). Returns the
    rerank shape: {"image_contains_fish": None, "rejection_reason": None,
    "observed_features": <dict|None>, "results": [...]}.
    """
    content = _request_deepseek([
        {"role": "system", "content": TEXT_RERANK_SYSTEM},
        {"role": "user", "content": build_user_prompt(caption, candidates)},
    ])
    return shape_results(caption, candidates, parse_json_object(content))
