import json

notebook_path = r"C:\Users\kaiso\PycharmProjects\BPAI\sanitychecks\thesis_analysis.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cell_source = [
    "# =================================================\n",
    "# 6. The \"Double Penalty\" Heatmap (Regressivity × Deprivation)\n",
    "# =================================================\n",
    "df = cached_data.get(\"socio_df_mapped_Baseline\")\n",
    "\n",
    "if df is not None and 'taxvaluedollarcnt' in df.columns and 'rounded_adi' in df.columns:\n",
    "    # Calculate Price Deciles\n",
    "    df['Price_Decile'] = pd.qcut(df['taxvaluedollarcnt'], 10, labels=False) + 1\n",
    "    \n",
    "    # Group by Price Decile and ADI\n",
    "    heatmap_data = df.groupby(['Price_Decile', 'rounded_adi'])['logerror'].mean().reset_index()\n",
    "    \n",
    "    # Pivot the data for the heatmap\n",
    "    heatmap_pivot = heatmap_data.pivot(index='Price_Decile', columns='rounded_adi', values='logerror')\n",
    "    \n",
    "    # Invert Y axis so Price Decile 10 (Most Expensive) is at the top, 1 (Cheapest) is at the bottom\n",
    "    heatmap_pivot = heatmap_pivot.sort_index(ascending=False)\n",
    "    \n",
    "    plt.figure(figsize=(15, 11))\n",
    "    \n",
    "    # Center the colormap at 0 (white), with red for overvaluation and blue for undervaluation\n",
    "    ax = sns.heatmap(heatmap_pivot, cmap='coolwarm', center=0, annot=True, fmt=\".3f\", \n",
    "                     annot_kws={\"size\": 14}, cbar_kws={'label': 'Mean Logerror (Positive = Overvalued)'})\n",
    "    \n",
    "    ax.set_xlabel('Area Deprivation Index (1 = Affluent, 10 = Deprived)', fontsize=24, labelpad=20)\n",
    "    ax.set_ylabel('Property Value Decile (10 = Most Expensive, 1 = Cheapest)', fontsize=24, labelpad=20)\n",
    "    ax.tick_params(axis='both', which='major', labelsize=20)\n",
    "    \n",
    "    # Heatmaps shouldn't have grid lines, ensuring they are off\n",
    "    ax.grid(False)\n",
    "        \n",
    "    plt.tight_layout(pad=2.0)\n",
    "    plt.savefig('double_penalty_heatmap.pdf', format='pdf', bbox_inches='tight')\n",
    "    plt.show()\n",
    "else:\n",
    "    print(\"Required data missing for Double Penalty Heatmap.\")\n"
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

print("Heatmap cell added.")
