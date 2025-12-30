import json

target_file = r"f:\ComputerVisionTestNew\notebooks\03_Evaluation_Combined.ipynb"

print(f"Reading {target_file}...")
with open(target_file, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = cell['source']
        if not source: continue
        first_line = source[0].strip()
        
        has_imports = any("from src.data.loader" in line for line in source)
        has_inline = any("def evaluate_model" in line for line in source)
        
        print(f"Cell {i}: start='{first_line[:20]}...' imports={has_imports} inline={has_inline}")
