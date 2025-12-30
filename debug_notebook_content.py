import json

target_file = r"f:\ComputerVisionTestNew\notebooks\03_Evaluation_Combined.ipynb"

print(f"Reading {target_file}...")
with open(target_file, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = cell['source']
        source_text = "".join(source)
        if "create_dataloaders" in source_text or "get_model" in source_text:
            print(f"--- Cell {i} Content Start ---")
            for line in source:
                print(line.rstrip())
            print(f"--- Cell {i} Content End ---")
