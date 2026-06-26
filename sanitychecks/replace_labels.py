import json

notebook_path = r"C:\Users\kaiso\PycharmProjects\BPAI\sanitychecks\thesis_analysis.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

replacements = {
    "ADI (1=Affluent, 10=Deprived)": "Median Area Deprivation Index (ADI)",
    "Ranked Neighborhoods (Affluent → Deprived)": "Cluster Rank (Ascending ADI)",
    "Cluster ID (Ordered by Increasing Deprivation)": "Socioeconomic Cluster Rank",
    "Cluster ID (Ordered by Increasing Deprivation": "Socioeconomic Cluster Rank", # Fallback for missing paren
    "Cluster ID (Affluent → Deprived)": "Socioeconomic Cluster Rank"
}

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            for old, new in replacements.items():
                if old in line:
                    line = line.replace(old, new)
            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Plot labels updated.")
