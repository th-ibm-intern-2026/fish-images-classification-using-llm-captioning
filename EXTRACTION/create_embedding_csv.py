# This script will be used to create a formatted CSV for embedding purposes.
import os
import pandas as pd
from physical_description_service import get_fish_description_from_watsonxai
# Placeholder for CSV creation logic

def create_embedding_csv(output_path):
    # Read fish names from the provided CSV
    import csv
    input_csv = "./DATA/fish-description-files/Marine_Fish_Species_Full_Description_test.csv"
    fish_names = []
    with open(input_csv, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            fish_names.append(row["Fish Name"])

    # get fish physical descriptions with checkpointing
    import json
    checkpoint_path = "fish_descriptions_checkpoint.json"
    # Try to load checkpoint if exists
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            fish_descriptions = json.load(f)
        print(f"Loaded checkpoint with {len(fish_descriptions)} fish descriptions.")
    else:
        fish_descriptions = {}

    for fish_name in fish_names:
        if fish_name in fish_descriptions and isinstance(fish_descriptions[fish_name], str) and fish_descriptions[fish_name]:
            # Already processed, skip
            print(f"Skipping {fish_name}, already processed.")
            continue
        try:
            description = get_fish_description_from_watsonxai(fish_name)
            fish_descriptions[fish_name] = description
        except Exception as e:
            print(f"Error getting description for {fish_name}: {e}")
            fish_descriptions[fish_name] = "body: , colors: , features: , unique_marks: "
        # Save checkpoint after each fish
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(fish_descriptions, f, ensure_ascii=False, indent=2)
        print(f"Processed {len(fish_descriptions)} out of {len(fish_names)} fish descriptions (checkpoint saved)")
        

    # Prepare new rows
    rows = []
    for fish in fish_names:
        # Format for COS object names
        fish_folder = fish.replace(" ", "-")
        fish_file = fish.lower().replace(" ", "-")
        object_names = [
            f"fish_images/{fish_folder}/{fish_file}-{i:03d}.png" for i in range(1, 4)
        ]
        rows.append({
            "Fish Name": fish,
            "Physical Description": fish_descriptions[fish],
            "Object Names": ", ".join(object_names)
        })

    # Write to new CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Example usage (to be updated with actual logic)
    create_embedding_csv("embedding_format.csv")
    



