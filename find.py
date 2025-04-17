import os
import json

json_file = "valid.json"
yes = 0
no = 0

# Read JSON file
with open(json_file, 'r') as f:
    data = json.load(f)

# Iterate through the JSON data
for template in data:
    if template.get('answer') and template['answer'][0].lower() == "yes":
        yes += 1
    else:
        no += 1

print(f"Yes is {yes} and No is {no} in {json_file}")