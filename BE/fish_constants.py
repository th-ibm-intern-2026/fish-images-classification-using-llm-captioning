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

Schema:
{{
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
    ]
}}
"""