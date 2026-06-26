"""Claude (Anthropic) vision provider for the fish image endpoints.

Mirrors the output shapes of the WatsonX/Groq/Gemini helpers so the Flask
routes can use Claude interchangeably:
  - caption_image_anthropic(b64)            -> str
  - identify_fish_details_anthropic(b64)    -> {"image_contains_fish", "fish_details"}
  - identify_fish_candidates_anthropic(b64) -> {"image_contains_fish", "rejection_reason", "results"}

The client is built lazily and only when ANTHROPIC_API_KEY is set, matching the
guarded-client pattern used for the Gemini/Groq providers in api_services.py.
"""
import base64
import json
import os

import anthropic

from fish_constants import SYSTEM_CONTENT_SINGLE, CAPTION_MATCH_SYSTEM, CAPTION_MATCH_USER
from text_rerank import TEXT_RERANK_SYSTEM, build_user_prompt, parse_json_object, shape_results

# Sonnet 4.6 — strong vision at a lower cost than Opus. Override with ANTHROPIC_MODEL.
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# Per-stage models for the two-pass /identify_and_search pipeline:
#   describe (Pass 1) -> Sonnet for accurate feature reading;
#   rerank  (Pass 2) -> Haiku, cheaper and sufficient to re-order a shortlist.
# Both fall back to MODEL and are overridable per-deployment.
CAPTION_MODEL = os.getenv("ANTHROPIC_CAPTION_MODEL", MODEL)
RERANK_MODEL = os.getenv("ANTHROPIC_RERANK_MODEL", "claude-haiku-4-5")
# Cheap up-front gate: a small Haiku call that only decides "is this a fish?" before
# the more expensive Sonnet caption runs (and before the text-only DeepSeek rerank,
# which never sees the image and so can't reject non-fish photos).
GATE_MODEL = os.getenv("ANTHROPIC_GATE_MODEL", "claude-haiku-4-5")

_client = None


def get_anthropic_client():
    """Return a cached Anthropic client, or None if no API key is configured."""
    global _client
    if _client is None:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            return None
        _client = anthropic.Anthropic(api_key=key)
    return _client


def _media_type(b64):
    """Sniff the image MIME type from the first few decoded bytes (default JPEG)."""
    try:
        head = base64.b64decode(b64[:32], validate=False)[:12]
    except Exception:
        return "image/jpeg"
    if head[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _image_block(b64):
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": _media_type(b64), "data": b64},
    }


def _cached_system(text):
    """Wrap a system prompt as a cacheable content block (Anthropic prompt caching).

    The big identifier prompt (all reference descriptions) is byte-stable across
    requests, so caching it serves it at ~0.1x input cost on warm hits. The image
    lives in the user message, after this breakpoint, so it never invalidates the
    cache. TTL is 5 min — effective once request traffic is steady.
    """
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def _response_text(resp):
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _parse_json_object(text):
    """Extract and parse the first {...} JSON object from the model's text output."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output: {text[:200]}")
    return json.loads(text[start : end + 1])


# Caption in the same 7-field template the ES physical_description embeddings use,
# so /identify_and_search matches like-for-like (see fish_constants.CAPTION_MATCH_SYSTEM).
_CAPTION_SYSTEM = CAPTION_MATCH_SYSTEM

_DETAILS_SYSTEM = """
You are an expert Ichthyologist and AI assistant specializing in marine biology and
taxonomy, particularly species found in Thailand. Analyze the image and return a
strictly formatted JSON response.

--- STEP 1: VALIDATION ---
Set "image_contains_fish" to false if the image shows: cooked food; processed fish
(fillets, dried, head removed); non-realistic images (cartoons, drawings); or is too
blurry to identify.

--- STEP 2: GENERATION ---
If valid, fill in the schema below.

--- OUTPUT SCHEMA ---
Return ONLY a raw JSON object (no markdown, no ```json fences):
{
    "image_contains_fish": <boolean>,
    "fish_details": {
        "fish_name": "<Common name in English>",
        "scientific_name": "<Scientific Latin name>",
        "order_name": "<Taxonomic Order>",
        "physical_description": "<3-5 sentences: body shape, scale patterns, coloration, fin characteristics, distinct anatomical features>",
        "habitat": "<environments, water depth, water type, behavior>"
    }
}
If "image_contains_fish" is false, "fish_details" must be an empty object {}.
"""


_FISH_GATE_SYSTEM = (
    "You are a strict image validator for a fish-identification pipeline. Look at the image "
    "and decide whether it shows a real, live/fresh/raw fish that could be identified to species "
    "from the photo. Answer false if the image shows: cooked or processed fish (fillets, dried, "
    "a cooked dish); a non-fish subject (another animal, a person, an object, scenery); a "
    "cartoon, drawing, or illustration; or is too blurry/dark/cropped to identify. "
    'Output ONLY JSON (no markdown): {"image_contains_fish": <true|false>, '
    '"rejection_reason": <short reason string, or null if it is a fish>}.'
)


def is_fish_image_anthropic(b64, model=None):
    """Cheap Haiku gate: does this image show an identifiable fish?

    Returns {"image_contains_fish": bool, "rejection_reason": str|None}. Meant to run
    before the Sonnet caption so non-fish images are rejected early and cheaply.
    """
    client = get_anthropic_client()
    resp = client.messages.create(
        model=model or GATE_MODEL,
        max_tokens=200,
        system=_FISH_GATE_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                _image_block(b64),
                {"type": "text", "text": "Is this an identifiable fish? Return the JSON object."},
            ],
        }],
    )
    return _parse_json_object(_response_text(resp))


def caption_image_anthropic(b64, model=None):
    client = get_anthropic_client()
    resp = client.messages.create(
        model=model or CAPTION_MODEL,
        max_tokens=900,
        system=_CAPTION_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                _image_block(b64),
                {"type": "text", "text": CAPTION_MATCH_USER},
            ],
        }],
    )
    return _response_text(resp)


def identify_fish_details_anthropic(b64):
    client = get_anthropic_client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1200,
        system=_DETAILS_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                _image_block(b64),
                {"type": "text", "text": "Identify the species and return the JSON object defined in your instructions."},
            ],
        }],
    )
    return _parse_json_object(_response_text(resp))


def rerank_candidates_haiku_text(caption, candidates, model=None):
    """TEXT-only Haiku fallback reranker (no image), mirroring the DeepSeek path.

    Matches the 7-field caption text against the candidate descriptions using the SAME
    shared prompt/shaping as rerank_candidates_deepseek, so the fallback is regularized
    with the primary path. Returns the shared text-rerank shape.
    """
    client = get_anthropic_client()
    resp = client.messages.create(
        model=model or RERANK_MODEL,
        max_tokens=2048,
        system=TEXT_RERANK_SYSTEM,
        messages=[{"role": "user", "content": build_user_prompt(caption, candidates)}],
    )
    return shape_results(caption, candidates, parse_json_object(_response_text(resp)))


def identify_fish_candidates_anthropic(b64):
    client = get_anthropic_client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=_cached_system(SYSTEM_CONTENT_SINGLE),  # cache the all-species prompt
        messages=[{
            "role": "user",
            "content": [
                _image_block(b64),
                {"type": "text", "text": "Identify the fish. Return JSON with the top 5 candidates per your schema."},
            ],
        }],
    )
    u = resp.usage
    print(f"[search_possible_fish] cache_read={getattr(u, 'cache_read_input_tokens', 0)} "
          f"cache_write={getattr(u, 'cache_creation_input_tokens', 0)} input={u.input_tokens}",
          flush=True)
    return _parse_json_object(_response_text(resp))
