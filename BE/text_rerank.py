"""Shared text-only rerank logic for /identify_and_search Pass 2.

Both rerankers (DeepSeek and the Haiku fallback) match the Sonnet 7-field caption
TEXT against the candidate reference descriptions TEXT — neither sees the image.
Keeping the prompt, candidate formatting, JSON parsing, and result shaping here means
the two paths behave identically; only the underlying LLM call differs.
"""
from fish_constants import PHYSICAL_FEATURE_FIELDS

# One prompt for every text reranker so their behaviour is regularized.
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


def parse_json_object(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in rerank output: {text[:200]}")
    import json
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
    None (text rerankers never see the image; the Haiku gate owns that verdict).
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
