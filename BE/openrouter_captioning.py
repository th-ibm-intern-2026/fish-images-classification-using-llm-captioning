import base64
import io
import os

import requests
from PIL import Image

from anthropic_captioning import _FISH_GATE_SYSTEM, _media_type, _parse_json_object

OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
# Qwen3-VL-8B-Instruct: best accuracy-per-dollar on the gate benchmark.
GATE_MODEL = os.getenv("OPENROUTER_GATE_MODEL", "qwen/qwen3-vl-8b-instruct")
# Same Qwen3-VL model does grounding (fish bounding box) for the optional crop.
CROP_MODEL = os.getenv("OPENROUTER_CROP_MODEL", "qwen/qwen3-vl-8b-instruct")
TIMEOUT = int(os.getenv("OPENROUTER_TIMEOUT", "60"))

# Crop tuning (see benchmarks/benchmark_qwen_crop.py for how these were chosen).
# Downscale to a known edge BEFORE asking for the box so we control the
# coordinate space and can map it back to the original exactly. Padding is
# generous on top because the dorsal fin is what the model trims most.
CROP_SEND_MAX_EDGE = int(os.getenv("OPENROUTER_CROP_SEND_MAX_EDGE", "1024"))
CROP_PAD_FRAC = float(os.getenv("OPENROUTER_CROP_PAD_FRAC", "0.10"))       # l/r/bottom
CROP_PAD_TOP_FRAC = float(os.getenv("OPENROUTER_CROP_PAD_TOP_FRAC", "0.20"))  # top
CROP_MIN_AREA_FRAC = float(os.getenv("OPENROUTER_CROP_MIN_AREA_FRAC", "0.02"))

# Cap the long edge of the image we send to the captioning vision model. The
# model downscales anything larger anyway (Sonnet 4.6 clamps to ~1568px /
# ~1.15MP before billing), so 1344 trims upload bytes AND billed vision tokens
# (~25% fewer than the 1568 ceiling) with negligible detail loss.
CAPTION_MAX_EDGE = int(os.getenv("CAPTION_MAX_EDGE", "1344"))

_CROP_SYSTEM = (
    "You are a precise object-localization model. You are given a photo. Find the single "
    "most prominent fish (the main subject) and return its bounding box. Use ABSOLUTE pixel "
    "coordinates with the origin at the TOP-LEFT corner of the image: x grows rightward, y "
    "grows downward. The box must enclose the WHOLE fish including fins and tail. "
    'Output ONLY one JSON object, no markdown, no prose: {"bbox_2d": [x1, y1, x2, y2]} '
    "where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. "
    'If there is no fish, output {"bbox_2d": null}.'
)


def _api_key():
    return os.getenv("OPENROUTER_API_KEY")


def is_fish_image_openrouter(b64, model=None):
    """Qwen3-VL gate: does this image show an identifiable fish?

    Returns {"image_contains_fish": bool, "rejection_reason": str|None}. Raises
    RuntimeError if no API key is configured; the caller treats any gate failure
    as non-fatal and continues to captioning.
    """
    key = _api_key()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    data_url = f"data:{_media_type(b64)};base64,{b64}"
    body = {
        "model": model or GATE_MODEL,
        "max_tokens": 200,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": _FISH_GATE_SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "Is this an identifiable fish? Return the JSON object."},
            ]},
        ],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    r = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=TIMEOUT)
    r.raise_for_status()
    content = (r.json().get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return _parse_json_object(content)


def _to_original_coords(bbox, sw, sh, ow, oh):
    """Map a Qwen3-VL box to ORIGINAL pixel coords.

    Qwen3-VL returns 0-1000 NORMALIZED coordinates (verified: it can return
    values larger than the sent image height, which only make sense as
    normalized). Because they're normalized, the box is independent of what size
    we sent -- we just scale by the original dimensions. We still handle the rare
    0-1 normalized form, and fall back to absolute sent-pixels only if a value
    exceeds 1000 (i.e. clearly not 0-1000 normalized -- a different provider)."""
    if not bbox or len(bbox) != 4:
        return None
    v = [float(x) for x in bbox]
    m = max(v)
    if m <= 1.001:                                       # 0-1 normalized
        return [v[0] * ow, v[1] * oh, v[2] * ow, v[3] * oh]
    if m <= 1001:                                        # 0-1000 normalized (Qwen3-VL)
        return [v[0] / 1000 * ow, v[1] / 1000 * oh, v[2] / 1000 * ow, v[3] / 1000 * oh]
    sx, sy = ow / sw, oh / sh                             # absolute sent pixels
    return [v[0] * sx, v[1] * sy, v[2] * sx, v[3] * sy]


def _padded_box(bbox, w, h):
    """Order, pad (extra on top), and clamp a box to the image. None if invalid
    or smaller than CROP_MIN_AREA_FRAC of the image (treated as a localization miss)."""
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * CROP_PAD_FRAC
    x2 += bw * CROP_PAD_FRAC
    y1 -= bh * CROP_PAD_TOP_FRAC
    y2 += bh * CROP_PAD_FRAC
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(round(x2))), min(h, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    if (x2 - x1) * (y2 - y1) < CROP_MIN_AREA_FRAC * w * h:
        return None
    return (x1, y1, x2, y2)


def locate_fish_bbox_qwen(b64, model=None):
    """Ask Qwen3-VL for the main fish's bounding box, returned in ORIGINAL pixel
    coords (or None if no fish / no box). Downscales to CROP_SEND_MAX_EDGE first
    so the returned coords map back to the original exactly. Raises RuntimeError
    if no API key; the caller treats any failure as "no box" and uses the full image."""
    key = _api_key()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    raw = base64.b64decode(b64)
    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB")
        ow, oh = im.size
        scale = min(1.0, CROP_SEND_MAX_EDGE / max(ow, oh)) if CROP_SEND_MAX_EDGE > 0 else 1.0
        if scale < 1.0:
            im = im.resize((max(1, int(ow * scale)), max(1, int(oh * scale))), Image.LANCZOS)
        sw, sh = im.size
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    body = {
        "model": model or CROP_MODEL,
        "max_tokens": 200,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": _CROP_SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "Return the bounding box of the main fish as JSON."},
            ]},
        ],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    r = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=TIMEOUT)
    r.raise_for_status()
    content = (r.json().get("choices") or [{}])[0].get("message", {}).get("content") or ""
    raw_bbox = _parse_json_object(content).get("bbox_2d")
    return _to_original_coords(raw_bbox, sw, sh, ow, oh)


def crop_b64_to_fish(b64, bbox):
    """Crop the ORIGINAL image to the padded fish box and return new base64 JPEG.
    Returns the UNCHANGED b64 if bbox is None or the box is implausibly small, so
    cropping can only ever help — a localization miss never drops the fish."""
    raw = base64.b64decode(b64)
    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB")
        box = _padded_box(bbox, im.width, im.height)
        if box is None:
            return b64
        buf = io.BytesIO()
        im.crop(box).save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def cap_image_b64(b64, max_edge=None):
    """Downscale a base64 image so its long edge is <= max_edge (default
    CAPTION_MAX_EDGE), returning new base64 JPEG. Downscale-ONLY: returns the
    UNCHANGED b64 if the image is already within the cap, can't be parsed, or
    re-encoding fails — so this can only ever shrink the payload and never breaks
    the caption path. Run AFTER cropping, right before the captioning call."""
    cap = CAPTION_MAX_EDGE if max_edge is None else max_edge
    if cap <= 0:
        return b64
    try:
        raw = base64.b64decode(b64)
        with Image.open(io.BytesIO(raw)) as im:
            if max(im.size) <= cap:
                return b64
            im = im.convert("RGB")
            im.thumbnail((cap, cap), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return b64
