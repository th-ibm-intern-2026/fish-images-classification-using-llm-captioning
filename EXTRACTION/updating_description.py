import csv
import json
import os
import io

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load JSON data (same directory as script)
json_path = os.path.join(script_dir, 'fish_descriptions_checkpoint.json')
with open(json_path, 'r', encoding='utf-8') as f_json:
    fish_desc = json.load(f_json)

# Read CSV and update Physical Description
input_csv = os.path.join(script_dir, 'DATA/fish-description-files/Marine_Fish_Species_Formatted.csv')
output_csv = os.path.join(script_dir, 'DATA/fish-description-files/Marine_Fish_Species_Formatted_updated.csv')

# Try common encodings for the input CSV
encodings_to_try = ['utf-8-sig', 'windows-1252', 'latin-1']
reader = None
f_in = None
for enc in encodings_to_try:
    try:
        f_in = open(input_csv, 'r', encoding=enc)
        reader = csv.DictReader(f_in)
        # Test read first row
        next(reader)
        f_in.seek(0)
        reader = csv.DictReader(f_in)
        print(f"Successfully opened CSV with encoding: {enc}")
        break
    except UnicodeDecodeError:
        if f_in:
            f_in.close()
        continue

if reader is None:
    print(f"Could not decode CSV file {input_csv} with any of {encodings_to_try}")
    exit(1)

with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f_out:
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()
    for row in reader:
        fish_name = row['Fish Name']
        if fish_name in fish_desc:
            row['Physical Description'] = fish_desc[fish_name]
        writer.writerow(row)

print("Physical Description column updated. Output saved to:", output_csv)


