import json

notebook_path = r"C:\Users\kaiso\PycharmProjects\BPAI\sanitychecks\thesis_analysis.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Remove the old heatmap cell
nb['cells'] = [cell for cell in nb['cells'] if "Double Penalty\" Heatmap" not in "".join(cell.get('source', []))]

new_cell_source = [
    "# =================================================\n",
    "# 6. Cluster Logerror Distributions (Ridgeline Plot)\n",
    "# =================================================\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "run_name = \"Weighted (ADI x2.0)\"\n",
    "df_geo = cached_data.get(f\"geo_gdf_mapped_{run_name}\")\n",
    "df_summary = cached_data.get(f\"geo_summary_{run_name}\")\n",
    "\n",
    "if df_geo is not None and df_summary is not None:\n",
    "    # Sort clusters by dominant ADI so the Y-axis goes from Affluent to Deprived\n",
    "    sorted_clusters = df_summary.sort_values('dominant_adi')['c'].astype(int).tolist()\n",
    "    \n",
    "    # Map cluster IDs to descriptive labels\n",
    "    cluster_labels = {}\n",
    "    for c in sorted_clusters:\n",
    "        adi_val = df_summary.loc[df_summary['c'] == c, 'dominant_adi'].values[0]\n",
    "        cluster_labels[c] = f\"C{int(c)} (ADI: {adi_val:.1f})\"\n",
    "        \n",
    "    df_geo_plot = df_geo.copy()\n",
    "    df_geo_plot['Cluster_Label'] = df_geo_plot['clusters'].map(cluster_labels)\n",
    "    \n",
    "    # Create the sorted label list for ordering the plot\n",
    "    ordered_labels = [cluster_labels[c] for c in sorted_clusters]\n",
    "    \n",
    "    # Filter outliers just for the visual density distribution (preserves aesthetic shape)\n",
    "    df_geo_plot = df_geo_plot[(df_geo_plot['logerror'] > -0.7) & (df_geo_plot['logerror'] < 0.7)]\n",
    "\n",
    "    # Set up the FacetGrid for the Ridgeline plot\n",
    "    sns.set_theme(style=\"white\", rc={\"axes.facecolor\": (0, 0, 0, 0)})\n",
    "    \n",
    "    # Custom palette that shifts from blue (affluent) to red (deprived)\n",
    "    pal = sns.color_palette(\"coolwarm\", len(ordered_labels))\n",
    "    \n",
    "    g = sns.FacetGrid(df_geo_plot, row=\"Cluster_Label\", row_order=ordered_labels, \n",
    "                      hue=\"Cluster_Label\", hue_order=ordered_labels, \n",
    "                      aspect=10, height=0.6, palette=pal)\n",
    "    \n",
    "    # Draw the densities\n",
    "    g.map(sns.kdeplot, \"logerror\", bw_adjust=.5, clip_on=False, fill=True, alpha=0.8, linewidth=1.5)\n",
    "    g.map(sns.kdeplot, \"logerror\", clip_on=False, color=\"w\", lw=2, bw_adjust=.5)\n",
    "    \n",
    "    # Draw the baseline\n",
    "    g.refline(y=0, linewidth=2, linestyle=\"-\", color=None, clip_on=False)\n",
    "    \n",
    "    # Add a vertical zero-error line across all facets\n",
    "    g.map(plt.axvline, x=0, color='black', linestyle='--', linewidth=1.5, zorder=0)\n",
    "\n",
    "    # Add the labels\n",
    "    def label_clusters(x, color, label):\n",
    "        ax = plt.gca()\n",
    "        ax.text(-0.02, .2, label, fontweight=\"bold\", color=color,\n",
    "                ha=\"right\", va=\"center\", transform=ax.transAxes, fontsize=14)\n",
    "                \n",
    "    g.map(label_clusters, \"logerror\")\n",
    "    \n",
    "    # Format the bottom axis\n",
    "    g.set_titles(\"\")\n",
    "    g.set(yticks=[], ylabel=\"\")\n",
    "    g.despine(bottom=True, left=True)\n",
    "    \n",
    "    # Adjust spacing to make them overlap (the \"ridgeline\" effect)\n",
    "    g.figure.subplots_adjust(hspace=-0.25)\n",
    "    \n",
    "    # Format the final X-axis\n",
    "    g.set_xlabels('Directional Logerror (Bias)', fontsize=22, labelpad=20)\n",
    "    for ax in g.axes.flat:\n",
    "        ax.tick_params(axis='x', which='major', labelsize=16)\n",
    "    \n",
    "    plt.savefig('cluster_ridgeline_plot.pdf', format='pdf', bbox_inches='tight')\n",
    "    plt.show()\n",
    "    \n",
    "    # Restore the default seaborn theme to not break other cells if re-run\n",
    "    sns.set_theme(style=\"whitegrid\")\n",
    "else:\n",
    "    print(f\"Required data missing for Ridgeline Plot ({run_name}).\")\n"
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

print("Ridgeline cell added.")
