"""Generate structured physical descriptions for fish species using Claude Haiku 4.5.

Each description follows a fixed 7-field template so it can be compared
field-by-field against what the vision model reads from an image (see
BE/fish_constants.py PHYSICAL_FEATURE_FIELDS / SYSTEM_CONTENT_SINGLE):

    body_shape: ...; primary_colors: ...; secondary_colors: ...; markings: ...;
    dorsal_fin: ...; caudal_fin: ...; unique_features: ...

Run as a script to (re)generate descriptions for every fish in the formatted
CSV, writing a resumable checkpoint JSON that updating_description.py consumes.
"""
import csv
import json
import os

from dotenv import load_dotenv
import anthropic

load_dotenv()

MODEL = os.getenv("DESCRIPTION_MODEL", "claude-sonnet-4-6")   # generator (Sonnet)
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-opus-4-8")      # Opus fact-checks/corrects each description

# The 7 fields must stay in lock-step with BE/fish_constants.py PHYSICAL_FEATURE_FIELDS.
SYSTEM_CONTENT = (
    "You are a marine biology expert specializing in fish species identification. "
    "Provide factual, accurate descriptions based on scientific knowledge. "
    "Answer only what is asked. Do not use markdown formatting. If a feature is "
    "not typically visible in a standard photograph, write 'not typically visible' "
    "for that field rather than speculating."
)

_USER_TEMPLATE = """Provide a visually precise description of the fish species '{fish_name}'.

Output EXACTLY one single line using the template below. Describe ONLY features that are clearly visible in a standard photograph of a live, fresh, or raw specimen. Do not include the species name, markdown, JSON, line breaks, or any filler text. Keep the field labels exactly as written.

body_shape: [one of: fusiform, laterally compressed, dorsoventrally flattened, elongated]; primary_colors: [dominant body colors]; secondary_colors: [accent colors on belly or highlights]; markings: [patterns like stripes, bars, or spots AND their exact location on the body]; dorsal_fin: [shape and distinct colors/patterns]; caudal_fin: [shape such as forked, rounded, squared, AND colors]; unique_features: [highly visible diagnostic markers such as a dark spot near the gill cover, a bar through the eye, or colored fin tips]

Describe {fish_name}."""


def get_fish_description(fish_name, client=None, model=MODEL):
    """Return the 7-field physical description string for one species."""
    client = client or anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    resp = client.messages.create(
        model=model,
        max_tokens=1000,
        system=SYSTEM_CONTENT,
        messages=[{"role": "user", "content": _USER_TEMPLATE.format(fish_name=fish_name)}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


# Backward-compat alias for create_embedding_csv.py (kept importing the old name).
get_fish_description_from_watsonxai = get_fish_description


_JUDGE_SYSTEM = (
    "You are a senior marine biologist fact-checking a candidate physical description of a "
    "fish species for a visual-identification database.\n"
    "The description MUST be ONE line of seven 'label: value' pairs separated by '; ', keeping "
    "these exact labels in this order, with NO species name, and ONLY features visible in a "
    "standard photo:\n"
    "body_shape: <...>; primary_colors: <...>; secondary_colors: <...>; markings: <...>; "
    "dorsal_fin: <...>; caudal_fin: <...>; unique_features: <...>\n"
    "Verify: (1) factual/visual accuracy for the named species (correct colors, body shape, "
    "diagnostic markings and fin shapes); (2) the format above — all seven labels present, in "
    "order, each as 'label: value'. KEEP the labels; never strip them. Only rewrite when the "
    "description is factually wrong/missing or the labeled format is broken; otherwise approve.\n"
    'Output ONLY JSON (no markdown): {"verdict": "approve" | "revise", "issues": "<short note or '
    'empty>", "corrected_description": "<full corrected one-line description WITH the seven '
    'labels, or empty string if approve>"}.'
)


def judge_fish_description(fish_name, description, client=None, model=JUDGE_MODEL):
    """Opus fact-checks one generated description.

    Returns {"verdict": "approve"|"revise", "issues": str, "corrected_description": str}.
    """
    client = client or anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=1200,
        system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": (
            f"Species: {fish_name}\nCandidate description:\n{description}\n\n"
            "Verify accuracy and template compliance; return the JSON object."
        )}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        raise ValueError(f"judge returned no JSON: {text[:160]}")
    return json.loads(text[s:e + 1])


def _fish_names(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        return [row["Fish Name"].strip() for row in csv.DictReader(f) if row.get("Fish Name", "").strip()]


def generate_all(csv_path=None, checkpoint_path=None, gen_model=MODEL, judge_model=JUDGE_MODEL, use_judge=True):
    """Generate each description (gen_model = Sonnet), then have the judge (Opus) fact-check
    and correct it. The FINAL (corrected-if-needed) description is what gets stored/embedded.
    Resumes from the checkpoint; a judge log records every verdict for transparency.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = csv_path or os.path.join(
        base_dir, "DATA", "fish-description-files", "Marine_Fish_Species_Formatted.csv"
    )
    checkpoint_path = checkpoint_path or os.path.join(base_dir, "fish_descriptions_checkpoint.json")
    log_path = os.path.join(base_dir, "fish_descriptions_judge_log.json")

    names = _fish_names(csv_path)
    descriptions, judge_log = {}, {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            descriptions = json.load(f)
        print(f"Loaded checkpoint with {len(descriptions)} existing descriptions.")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            judge_log = json.load(f)

    client = anthropic.Anthropic()
    for i, name in enumerate(names, 1):
        if isinstance(descriptions.get(name), str) and descriptions[name].strip():
            print(f"[{i}/{len(names)}] skip (cached): {name}")
            continue
        try:
            raw = get_fish_description(name, client=client, model=gen_model)
            final, verdict, issues = raw, "skipped", ""
            if use_judge:
                v = judge_fish_description(name, raw, client=client, model=judge_model)
                verdict = (v.get("verdict") or "").lower()
                issues = v.get("issues") or ""
                corrected = (v.get("corrected_description") or "").strip()
                if verdict == "revise" and corrected:
                    final = corrected
            descriptions[name] = final
            judge_log[name] = {"verdict": verdict, "issues": issues, "generated": raw, "final": final}
            print(f"[{i}/{len(names)}] {name}: judge={verdict}{' (revised)' if final != raw else ''}")
        except Exception as e:
            print(f"[{i}/{len(names)}] ERROR {name}: {e}")
            continue
        # checkpoint after each success so a crash never loses progress
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(descriptions, f, ensure_ascii=False, indent=2)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(judge_log, f, ensure_ascii=False, indent=2)

    revised = sum(1 for v in judge_log.values() if v.get("verdict") == "revise")
    print(f"Done. {len(descriptions)} descriptions saved. Judge revised {revised}/{len(judge_log)}. Log: {log_path}")
    return descriptions


if __name__ == "__main__":
    generate_all()
