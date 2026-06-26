import json
import re

notebook_path = r"C:\Users\kaiso\PycharmProjects\BPAI\sanitychecks\thesis_analysis.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Remove rotation kwargs
            line = re.sub(r",\s*rotation=\d+", "", line)
            line = re.sub(r"rotation=\d+,\s*", "", line)
            # Remove ha='right' kwargs
            line = re.sub(r",\s*ha='right'", "", line)
            line = re.sub(r"ha='right',\s*", "", line)
            
            # Remove tick_params with only rotation
            if "ax.tick_params(axis='x', rotation=45)" in line:
                continue # Just delete the line completely
                
            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Rotations removed.")
