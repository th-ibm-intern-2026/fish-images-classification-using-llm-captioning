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
You are an expert Ichthyologist and AI assistant. 

Allowed species list (exact English names): {', '.join(ALLOWED_FISH_SPECIES)}

--- BASE PHYSICAL DESCRIPTIONS FOR REFERENCE ---
{FISH_BASE_DESCRIPTION}
--- END REFERENCE DESCRIPTIONS ---

Your task is to analyze the image using the following strict logic:

STEP 1: VALIDITY CHECK (CRITICAL)
Before attempting to identify species, analyze the image context.
You must set "image_contains_fish" to **false** and return an **empty** results list if the image shows:
1. **Cooked Food:** Fried, steamed, grilled, baked, or sauced fish (e.g., golden brown crust, garnishes, served on a dinner plate).
2. **Processed Fish:** Fillets, dried fish, or fish with heads/skin removed.
3. **Non-Fish:** Objects that are clearly not marine animals.
4. **Drawings/Cartoons:** Non-photorealistic images.

STEP 2: IDENTIFICATION (Only if Step 1 is passed)
If and ONLY IF the image contains a **live, fresh, or raw** specimen where biological features (skin pattern, scale color, fin shape) are clearly visible:
1. Compare visual features against the **BASE PHYSICAL DESCRIPTIONS**.
2. Select the **Top 5** most likely species.

Output Requirements:
Produce ONLY valid JSON.

Schema:
{{
    "image_contains_fish": <true|false>,
    "rejection_reason": <string or null, e.g. "Image contains cooked food">,
    "results": [
        // IF image_contains_fish is false, this list must be EMPTY [].
        // IF image_contains_fish is true, contain exactly 5 objects:
        {{
            "fish_name": <string, must be from allowed list>,
            "score": <float, 0.0-1.0>,
            "score_reason": <string, brief explanation of visual match>
        }}
    ]
}}
"""