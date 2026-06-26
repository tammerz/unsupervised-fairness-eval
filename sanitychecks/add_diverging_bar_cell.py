import json

notebook_path = r"C:\Users\kaiso\PycharmProjects\BPAI\sanitychecks\thesis_analysis.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Remove the old ridgeline plot cell
nb['cells'] = [cell for cell in nb['cells'] if "Ridgeline Plot" not in "".join(cell.get('source', []))]

new_cell_source = [
    "# =================================================\n",
    "# 6. Diverging Bar Chart (Directional Harm by Cluster)\n",
    "# =================================================\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "run_name = \"Weighted (ADI x2.0)\"\n",
    "df_summary = cached_data.get(f\"geo_summary_{run_name}\")\n",
    "\n",
    "if df_summary is not None:\n",
    "    # Sort clusters by ADI descending, so the highest ADI (10) gets plotted at the bottom of the Y-axis,\n",
    "    # and the lowest ADI (1) gets plotted at the top.\n",
    "    df_plot = df_summary.copy()\n",
    "    df_plot = df_plot.sort_values(by=['dominant_adi', 'logerror_mean'], ascending=[False, True])\n",
    "    \n",
    "    # Create descriptive labels\n",
    "    df_plot['Cluster_Label'] = df_plot.apply(lambda row: f\"C{int(row['c'])} (ADI: {row['dominant_adi']:.1f})\", axis=1)\n",
    "    \n",
    "    # Assign colors: Vermilion (Red) for overvaluation (>0), Sky Blue for undervaluation (<0)\n",
    "    # These are part of the Okabe-Ito colorblind-safe palette\n",
    "    df_plot['Color'] = df_plot['logerror_mean'].apply(lambda x: \"#D55E00\" if x > 0 else \"#56B4E9\")\n",
    "    \n",
    "    plt.figure(figsize=(14, 16))\n",
    "    ax = plt.gca()\n",
    "    \n",
    "    # Plot the diverging bars\n",
    "    bars = ax.barh(df_plot['Cluster_Label'], df_plot['logerror_mean'], \n",
    "                   color=df_plot['Color'], edgecolor='white', height=0.75, zorder=4)\n",
    "    \n",
    "    # Draw the thick black baseline at 0\n",
    "    ax.axvline(0, color='black', linewidth=2.5, linestyle='-', zorder=5)\n",
    "    \n",
    "    # Add data labels at the end of each bar for precise readability\n",
    "    for bar in bars:\n",
    "        width = bar.get_width()\n",
    "        label_x_pos = width + 0.001 if width > 0 else width - 0.001\n",
    "        ha = 'left' if width > 0 else 'right'\n",
    "        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f\"{width:+.3f}\",\n",
    "                ha=ha, va='center', fontsize=12, fontweight='bold', color=bar.get_facecolor())\n",
    "\n",
    "    # Titles and Labels\n",
    "    ax.set_xlabel('Mean Directional Logerror (Bias)', fontsize=24, labelpad=25)\n",
    "    ax.set_ylabel('Geographic Clusters (Affluent → Deprived)', fontsize=24, labelpad=25)\n",
    "    \n",
    "    ax.tick_params(axis='y', which='major', labelsize=16)\n",
    "    ax.tick_params(axis='x', which='major', labelsize=20)\n",
    "    \n",
    "    # Apply formatting\n",
    "    if 'enforce_academic_formatting' in globals():\n",
    "        enforce_academic_formatting(ax)\n",
    "        \n",
    "    plt.tight_layout(pad=2.0)\n",
    "    plt.savefig('diverging_directional_harm.pdf', format='pdf', bbox_inches='tight')\n",
    "    plt.show()\n",
    "else:\n",
    "    print(f\"Required data missing for Diverging Bar Chart ({run_name}).\")\n"
]

new_cell = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": new_cell_source
}

nb['cells'].append(new_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Diverging Bar Chart cell added.")
