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

from fish_constants import SYSTEM_CONTENT_OPEN, SYSTEM_CONTENT_DETAILS, CAPTION_MATCH_SYSTEM, CAPTION_MATCH_USER

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# Pass-1 caption model for /identify_and_search; falls back to MODEL, overridable.
CAPTION_MODEL = os.getenv("ANTHROPIC_CAPTION_MODEL", MODEL)

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

_DETAILS_SYSTEM = SYSTEM_CONTENT_DETAILS


_FISH_GATE_SYSTEM = (
    "You are a strict image validator for a fish-identification pipeline. Look at the image "
    "and decide whether it shows a real, live/fresh/raw fish that could be identified to species "
    "from the photo. Answer false if the image shows: cooked or processed fish (fillets, dried, "
    "a cooked dish); a non-fish subject (another animal, a person, an object, scenery); a "
    "cartoon, drawing, or illustration; or is too blurry/dark/cropped to identify. "
    'Output ONLY JSON (no markdown): {"image_contains_fish": <true|false>, '
    '"rejection_reason": <short reason string, or null if it is a fish>}.'
)


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


def identify_fish_candidates_anthropic(b64):
    client = get_anthropic_client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=_cached_system(SYSTEM_CONTENT_OPEN),  # cache the open identifier prompt
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
