import json

target_file = r"f:\ComputerVisionTestNew\notebooks\03_Evaluation_Combined.ipynb"

print(f"Processing {target_file}...")
try:
    with open(target_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    updated = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            
            # Check if this is the cell with inlined code (eval function)
            has_inline_defs = False
            for line in source:
                if "def evaluate_model" in line:
                    has_inline_defs = True
                    break
            
            if has_inline_defs:
                print("Found target cell.")
                new_source = []
                cell_changed = False
                for line in source:
                    # Remove the src imports
                    if line.strip().startswith("from src."):
                        print(f"Removing line: {line.strip()}")
                        cell_changed = True
                        continue
                    new_source.append(line)
                
                if cell_changed:
                    cell['source'] = new_source
                    updated = True
                    break # Assuming only one such cell
    
    if updated:
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        print(f"Successfully cleaned up {target_file}")
    else:
        print("No lines removed.")

except Exception as e:
    print(f"Error processing {target_file}: {e}")
