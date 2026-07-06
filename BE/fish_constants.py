# --- Configuration ---
MODEL_ID = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

# --- Shared structured-feature template ---
# These are the exact fields EXTRACTION/physical_description_service.py uses to author
# every species' "Physical Description" in the reference data. The caption step reads
# the image with the SAME fields so the embedding match is like-for-like.
PHYSICAL_FEATURE_FIELDS = [
    "body_shape",
    "primary_colors",
    "secondary_colors",
    "markings",
    "dorsal_fin",
    "caudal_fin",
    "unique_features",
]

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


# --- System Prompt ---
SYSTEM_CONTENT_SINGLE = f"""
You are an expert marine biologist with comprehensive knowledge of ALL aquatic and marine organisms.

--- KNOWN SPECIES DESCRIPTIONS FOR REFERENCE (use for visual matching guidance, not as a hard limit) ---
{FISH_BASE_DESCRIPTION}
--- END REFERENCE DESCRIPTIONS ---

STEP 1: DECIDE — set "image_contains_fish" to true or false.

✅ Set "image_contains_fish" to TRUE for ANY identifiable sea animal, including but not limited to:
  - Fish (all species, including sharks, rays, eels, seahorses)
  - Jellyfish & sea anemones
  - Octopus & squid (cephalopods)
  - Turtle & sea snakes (marine reptiles)
  - Lobster, crab, shrimp (crustaceans)
  - Oyster, clam, mussel, scallop, shell (bivalves & mollusks)
  - Starfish, sea urchin, sea cucumber (echinoderms)
  - Whale, dolphin, seal (marine mammals)
  - Prehistoric or extinct aquatic species
  - 3D renders, illustrations, fossils, or reconstructions of any of the above

❌ Set "image_contains_fish" to FALSE (and return empty results) ONLY for:
  - **Cooked/processed food:** Organism already prepared for eating (fried, grilled, filleted, served on a plate)
  - **Stone/hard coral:** Sessile colonial organisms like brain coral, staghorn coral, table coral — with NO identifiable animals present
  - **Marine plants:** Seagrass, kelp, algae, mangroves — with NO identifiable animals present
  - **Completely non-animal:** Rocks, sand, water, people, boats, etc. — with NO identifiable animals present
  - **Unidentifiable:** Too blurry, abstract, or obscured to determine any species

STEP 2: IDENTIFICATION (only if image_contains_fish is true)
1. Use your full marine biology knowledge — NOT limited to the reference list above.
2. Identify the top 5 most likely species using ALL visible features.
3. Output the **scientific name** for each (e.g. "Octopus vulgaris", "Chelonia mydas", "Panulirus argus").

Output Requirements:
Produce ONLY valid JSON. No markdown fences.
=======
# --- Details identification prompt (/image_identification) ---
# Single shared prompt for the per-provider details functions (anthropic/gemini/groq/
# watsonx) so they all return the SAME {image_contains_fish, fish_details} shape and
# the SAME language. Structured wording follows the Anthropic version; the
# physical_description and habitat fields are written in THAI for the Thai-facing app
# (names/scientific/order stay English).
SYSTEM_CONTENT_DETAILS = """
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
        "physical_description": "<3-5 sentences IN THAI: body shape, scale patterns, coloration, fin characteristics, distinct anatomical features>",
        "habitat": "<IN THAI: environments, water depth, water type, behavior>"
    }
}
If "image_contains_fish" is false, "fish_details" must be an empty object {}.
"""

# --- Open-ended candidates prompt (/search_possible_fish + identify_and_search fallback) ---
# Used when a fish is likely NOT one of the in-database species, so the model makes a
# free, open-ended guess. NOT constrained to any allowed list — it names the actual
# species it believes the fish is.
SYSTEM_CONTENT_OPEN = """
You are an expert Ichthyologist and AI assistant specializing in marine biology and
taxonomy, particularly species found in Thailand. Analyze the image and return a
strictly formatted JSON response with your best open-ended species guesses.

--- STEP 1: VALIDITY CHECK ---
Set "image_contains_fish" to false (and return an EMPTY results list) if the image shows:
1. Cooked or processed fish (fried, steamed, grilled, fillets, dried, head/skin removed).
2. Non-fish subjects (other animals, people, objects, scenery).
3. Drawings, cartoons, or other non-photorealistic images.
4. Images too blurry, dark, or cropped to identify.

--- STEP 2: IDENTIFICATION ---
If and ONLY IF the image shows a live, fresh, or raw fish, identify it. You are NOT
limited to any predefined list — name the actual species you believe it is, using its
real common name. Give your Top 5 most likely species, ordered most-likely first.

--- OUTPUT SCHEMA ---
Return ONLY a raw JSON object (no markdown, no ```json fences):
{
    "image_contains_fish": <true|false>,
    "rejection_reason": <string or null, reason if false>,
    "results": [
        // Empty [] if image_contains_fish is false.
        // Exactly 5 objects if image_contains_fish is true:
        {{
            "fish_name": <string, SCIENTIFIC NAME e.g. "Octopus vulgaris">,
            "score": <float 0.0-1.0>,
            "score_reason": <string, brief visual explanation>
        }}
=======
        {
            "fish_name": <string, common name of the species>,
            "score": <float, 0.0-1.0, your confidence>,
            "score_reason": <string, the visible features that led to this guess>
        }
    ]
}
If "image_contains_fish" is false, "results" must be an empty list [].
"""