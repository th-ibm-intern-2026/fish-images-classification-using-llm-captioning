import csv
import os

# --- Configuration ---
MODEL_ID = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
# MODEL_ID =  "meta-llama/llama-3-2-90b-vision-instruct";
CSV_FILENAME = "Marine_Fish_Possible_Output.csv"

def load_fish_data_from_csv():
    """
    Reads the CSV file from the same directory and returns:
    1. A list of allowed species names.
    2. A formatted text description string.
    """
    allowed_species = []
    description_lines = []
    
    # Locate CSV file relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, CSV_FILENAME)

    if not os.path.exists(file_path):
        # Fallback if file is missing (to prevent crash on import)
        print(f"⚠️ Warning: '{CSV_FILENAME}' not found in {base_dir}")
        return [], ""

    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Check for required columns
            if 'Fish Name' not in reader.fieldnames or 'Physical Description' not in reader.fieldnames:
                print("⚠️ Error: CSV is missing 'Fish Name' or 'Physical Description' columns.")
                return [], ""

            for row in reader:
                name = row['Fish Name'].strip()
                desc = row['Physical Description'].strip()
                
                if name and desc:
                    allowed_species.append(name)
                    # Format: "FishName Description"
                    description_lines.append(f"{name} {desc}")
                    
        return allowed_species, "\n".join(description_lines)

    except Exception as e:
        print(f"⚠️ Error loading fish constants: {e}")
        return [], ""

# --- Load Data on Module Import ---
ALLOWED_FISH_SPECIES, FISH_BASE_DESCRIPTION = load_fish_data_from_csv()

# --- Shared structured-feature template ---
# These are the exact fields EXTRACTION/physical_description_service.py uses to author
# every species' "Physical Description" (which becomes FISH_BASE_DESCRIPTION above).
# The identifier reads the image with the SAME fields so matching is like-for-like.
PHYSICAL_FEATURE_FIELDS = [
    "body_shape",
    "primary_colors",
    "secondary_colors",
    "markings",
    "dorsal_fin",
    "caudal_fin",
    "unique_features",
]

# Human-readable spec for each field, embedded into the system prompt and reused
# (via PHYSICAL_FEATURE_FIELDS) to build the response schema in fish_services.py.
_FEATURE_GUIDE = """   - body_shape: one of [fusiform, laterally compressed, dorsoventrally flattened, elongated]
   - primary_colors: dominant body colors
   - secondary_colors: accent colors on belly or highlights
   - markings: patterns (stripes, bars, spots) AND their exact location on the body
   - dorsal_fin: shape and distinct colors/patterns
   - caudal_fin: shape (forked, rounded, squared) and colors
   - unique_features: highly visible diagnostic markers (e.g. dark spot near the gill cover, bar through the eye, colored fin tips)"""

# --- Caption prompt for /identify_and_search (image -> caption -> embed -> ES) ---
# The ES physical_description_embedding is now built from 7-field descriptions
# (EXTRACTION/physical_description_service.py). Emitting the caption in the SAME
# 7-field shape (no species name) makes the vector match like-for-like instead of
# matching free prose against structured text.
CAPTION_MATCH_SYSTEM = (
    "You are an expert ichthyologist describing a fish from a photo for visual "
    "database matching. Describe ONLY what is clearly visible. Do not name the "
    "species. Do not use markdown, JSON, or line breaks. Output EXACTLY one line "
    "using this template, keeping the field labels:\n"
    "body_shape: <fusiform, laterally compressed, dorsoventrally flattened, or elongated>; "
    "primary_colors: <dominant body colors>; secondary_colors: <accent colors on belly or highlights>; "
    "markings: <patterns like stripes, bars, or spots AND their location on the body>; "
    "dorsal_fin: <shape and colors/patterns>; caudal_fin: <shape such as forked, rounded, squared, AND colors>; "
    "unique_features: <highly visible diagnostic markers>\n"
    'Write "not clearly visible" for any field the photo does not show. Do not invent sizes or measurements.'
)
CAPTION_MATCH_USER = "Describe the fish in this image using the template."

# --- Rerank prompt for /identify_and_search (image + ES candidates -> LLM re-score) ---
# After the vector search returns its top-N shortlist, we hand the ORIGINAL image
# back to the vision model together with each shortlisted species' reference
# description, and let the model re-score them by actually looking at the photo.
# This fixes the weakness of pure kNN, where scores bunch together (~0.88-0.92)
# and the true species often is not ranked #1 even when it is in the shortlist.
RERANK_SYSTEM = f"""
You are an expert ichthyologist. You are given a photo of a fish and a SHORTLIST of
candidate species (each with a reference physical description authored with a fixed
7-field template). The shortlist came from an automated vector search and may be
mis-ordered. Re-rank ONLY these candidates by how well each matches the fish actually
shown in the photo, using the SAME structured logic an expert would.

STEP 1: VALIDITY CHECK
If the photo is not a live/fresh/raw identifiable fish (cooked, processed, a drawing,
or too blurry to read features), set "image_contains_fish" to false and return an
empty results list.

STEP 2: OBSERVE
Read the specimen in the photo using this exact field template, and record it in
"observed_features":
{_FEATURE_GUIDE}
   Use "not visible" for any field the photo does not clearly show. Never guess.

STEP 3: COMPARE & SCORE
For EACH candidate, match your observed_features against that candidate's reference
description field-by-field. Weight agreement in unique_features and markings most
heavily; a single matching color must NOT outweigh a clear mismatch in body_shape or
markings. Set "rerank_score" (0.0-1.0) from how many fields agree and how diagnostic
they are. In "rerank_reason", name the specific fields that matched or conflicted
(e.g. "markings and caudal_fin match; body_shape differs").

Rules:
- Choose ONLY from the candidates given. Do not introduce other species.
- Return EXACTLY one result object per candidate, ordered best-match first.

Output ONLY valid JSON (no markdown, no ```json fences):
{{
    "image_contains_fish": <true|false>,
    "rejection_reason": <string or null>,
    "observed_features": {{
        "body_shape": <string>, "primary_colors": <string>, "secondary_colors": <string>,
        "markings": <string>, "dorsal_fin": <string>, "caudal_fin": <string>,
        "unique_features": <string>
    }},
    "results": [
        {{
            "fish_name": <string, must equal one of the candidate names>,
            "scientific_name": <string, copy from the candidate>,
            "rerank_score": <float 0.0-1.0, your visual-match confidence>,
            "rerank_reason": <string, name the fields that matched or conflicted>
        }}
    ]
}}
"""

# --- System Prompt ---
SYSTEM_CONTENT_SINGLE = f"""
You are an expert Ichthyologist and AI assistant.

Allowed species list (exact English names): {', '.join(ALLOWED_FISH_SPECIES)}

--- BASE PHYSICAL DESCRIPTIONS FOR REFERENCE ---
{FISH_BASE_DESCRIPTION}
--- END REFERENCE DESCRIPTIONS ---

Each reference description above was authored with this exact field template:
<name>; body_shape: ...; primary_colors: ...; secondary_colors: ...; markings: ...; dorsal_fin: ...; caudal_fin: ...; unique_features: ...

Your task is to analyze the image using the following strict logic:

STEP 1: VALIDITY CHECK (CRITICAL)
Before attempting to identify species, analyze the image context.
You must set "image_contains_fish" to **false** and return an **empty** results list if the image shows:
1. **Cooked Food:** Fried, steamed, grilled, baked, or sauced fish (e.g., golden brown crust, garnishes, served on a dinner plate).
2. **Processed Fish:** Fillets, dried fish, or fish with heads/skin removed.
3. **Non-Fish:** Objects that are clearly not marine animals.
4. **Drawings/Cartoons:** Non-photorealistic images.

STEP 2: STRUCTURED IDENTIFICATION (Only if Step 1 is passed)
If and ONLY IF the image contains a **live, fresh, or raw** specimen where biological features (skin pattern, scale color, fin shape) are clearly visible:

1. OBSERVE — read the specimen in the image using the SAME field template as the reference descriptions, and record it in "observed_features":
{_FEATURE_GUIDE}
   Use "not visible" for any field the photo does not clearly show. Never guess.

2. COMPARE — for each candidate species, match your observed_features against that species' reference description field-by-field. Weight agreement in unique_features and markings most heavily; a single matching color must NOT outweigh a clear mismatch in body_shape or markings.

3. SCORE — set "score" from how many fields agree and how diagnostic they are. In "score_reason", name the specific fields that matched and any that conflicted (e.g. "markings and caudal_fin match; body_shape differs").

4. Select the **Top 5** most likely species from the allowed list.

Output Requirements:
Produce ONLY valid JSON.

Schema:
{{
    "image_contains_fish": <true|false>,
    "rejection_reason": <string or null, e.g. "Image contains cooked food">,
    "observed_features": {{
        // The specimen as read from the image, in the reference template.
        // May be an empty object {{}} if image_contains_fish is false.
        "body_shape": <string>,
        "primary_colors": <string>,
        "secondary_colors": <string>,
        "markings": <string>,
        "dorsal_fin": <string>,
        "caudal_fin": <string>,
        "unique_features": <string>
    }},
    "results": [
        // IF image_contains_fish is false, this list must be EMPTY [].
        // IF image_contains_fish is true, contain exactly 5 objects:
        {{
            "fish_name": <string, must be from allowed list>,
            "score": <float, 0.0-1.0>,
            "score_reason": <string, name the observed_features fields that matched or conflicted>
        }}
    ]
}}
"""