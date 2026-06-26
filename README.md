# C4F (Clustering for Fairness)

C4F is an open-source tool for **evaluating machine learning model fairness without ground-truth labels**.

Standard fairness metrics (like demographic parity or equal opportunity) require either true labels or predefined thresholds. In many real-world deployments (e.g., credit scoring, hiring, medical diagnosis), ground truth is delayed or unavailable, and setting arbitrary thresholds masks how errors are actually distributed.

C4F solves this by using **unsupervised clustering** on the model's feature space, combined with the model's errors (or predictions) and sensitive attributes. It identifies distinct subgroups where the model behaves differently and highlights disparate error rates across demographic groups.

## How it works

1. **Cluster:** Group instances using features (regular, sensitive, and special like SHAP values).
2. **Evaluate:** For each cluster, compute the error rate and demographic composition.
3. **Compare:** Use statistical tests (Chi-squared, Kruskal-Wallis, Mann-Whitney U) to find clusters with significantly higher/lower errors and check if certain demographic groups are over-represented in those clusters.

This allows you to say: *"The model has a significantly higher error rate on this specific cluster of users, and this cluster is disproportionately composed of older females."*

## Key Features

- **Multiple Algorithms:** HDBSCAN (default, robust to noise), DBSCAN, K-Means, Bisecting K-Means, K-Medoids, and K-Prototypes.
- **Distance Metrics:** Euclidean, Manhattan, and Gower (for mixed data types).
- **Automated k-selection:** Optimize the number of clusters using Silhouette score, error separability, or a composite metric.
- **Regression & Classification:** Supports both continuous error values (regression) and binary errors (classification).
- **Multi-class Sensitive Attributes:** Automatically handles sensitive attributes with 3+ categories (e.g., race, location) by expanding them into binary indicators for per-group fairness analysis.
- **Automated Encoding:** Categorical features are automatically one-hot encoded (or handled natively by K-Prototypes).
- **Web Interface:** Includes a Flask-based web UI for interactive analysis and visualization.
- **Batch Experiment Mode:** Automatically test combinations of feature groups (Regular, Sensitive, Special) to see how different representations impact fairness detection.

---

## Installation

### As a standalone tool

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-org/BPAI.git
cd BPAI
pip install -r requirements.txt
```

### As a Python package

You can install C4F directly into your environment to use it programmatically:

```bash
pip install .
```

*Note: The package name is `c4f`.*

---

## Usage

### 1. Web UI (Recommended)

The easiest way to use C4F is through the interactive web interface.

```bash
python webapp/app.py
```

Then open `http://localhost:5050` in your browser.
1. Upload your CSV dataset.
2. Assign roles to your columns (Regular feature, Sensitive feature, Error column).
3. Configure parameters (Algorithm, Distance, Run mode).
4. Run the analysis and explore the interactive heatmaps and scatter plots.

### 2. Command Line Interface (CLI)

You can run C4F from the command line for automation or batch processing.

#### Basic Single Run

```bash
python main.py \
  --data_path Data/your_dataset.csv \
  --regular_cols "age,income,credit_history" \
  --sensitive_cols "gender,race" \
  --error_col "prediction_error" \
  --error_type "binary" \
  --algorithm hdbscan \
  --min_datapoints 20
```

#### Range Search for `k` (e.g., K-Means)

```bash
python main.py \
  --data_path Data/your_dataset.csv \
  --regular_cols "f1,f2,f3" \
  --sensitive_cols "sensitive_attr" \
  --error_col "error" \
  --algorithm kmeans \
  --n_min 3 \
  --n_max 8 \
  --scoring composite
```

#### Batch Experiment Mode

Automatically run clustering on different combinations of feature groups (e.g., Regular only, Regular + Sensitive, Regular + Special).

```bash
python main.py \
  --data_path Data/your_dataset.csv \
  --regular_cols "f1,f2" \
  --sensitive_cols "s1" \
  --special_cols "shap1,shap2" \
  --error_col "error" \
  --experiment \
  --algorithm hdbscan
```

*(You can exclude specific groups from the combinations using `--experiment SPECIAL`, which removes the `SPECIAL` group from the clustering combinations but keeps it for fairness evaluation).*

### 3. Python API

You can import C4F directly into your Python scripts or Jupyter Notebooks.

```python
import pandas as pd
from c4f.clustering import cluster
from c4f.experiments import make_recap, separability_check

# Load data
df = pd.read_csv("Data/your_dataset.csv")

# 1. Run clustering
result = cluster(
    features=df[['f1', 'f2', 's1']],
    algorithm='hdbscan',
    min_cluster_size=15
)

# 2. Add labels to DataFrame
df['clusters'] = result.labels

# 3. Generate fairness recap
recap = make_recap(
    data_result=df,
    feature_set=['f1', 'f2', 's1'],
    sensitive_cols=['s1'],
    error_col='error',
    error_type='binary',
    feature_matrix=result.feature_matrix
)
print(recap)

# 4. Statistical separability check
stats = separability_check(df, result.labels, columns=['f1', 'f2', 's1'])
print(stats)
```

---

## Output Files

When running via the CLI, C4F creates a timestamped folder in `clustering_results/` containing:

**Single Run Mode:**
- `recap/<run_name>.csv`: Detailed cluster statistics (size, error rate, demographic proportions, silhouette score).
- `separability/<run_name>.csv`: Kruskal-Wallis / Chi-squared p-values indicating if features differ significantly across clusters.
- `recap/<run_name>.svg`: Heatmap visualization of the recap table.
- `clusters.svg`: 2D projection (t-SNE, PCA, or MDS) scatter plot colored by cluster.
- `composition_<attr>.svg`: Stacked bar charts showing the demographic composition of each cluster.

**Experiment Mode:**
- `results_summary.csv`: Global overview of all conditions (1 row per condition), highlighting the error Kruskal-Wallis p-value and max error difference.
- `<condition_name>.csv`: Detailed per-cluster recap for a specific feature combination, including a "rule" column describing the most distinctive trait of each cluster.
- `chi_res.csv` / `chi_res_heatmap.svg`: Statistical test p-values for error and sensitive attributes across all conditions.
- `all_quali_heatmap.svg`: Overview heatmap combining statistical significance with clustering quality (Silhouette) across conditions.
- `exp_condition.csv`: Reference table of the evaluated feature combinations.
- `<condition_name>_clusters.svg` & `<condition_name>_composition_<attr>.svg`: Visualizations for each condition.

---

## Configuration Reference

### Algorithms & Distance
- `--algorithm`: `hdbscan`, `dbscan`, `kmeans`, `bisectingkmeans`, `kmedoids`, `kprototypes`.
- `--distance`: `euclidean`, `manhattan`, `gower`. *(Note: K-Prototypes uses its own metric; Gower distance triggers MDS for projections and custom silhouette computation).*

### Error Types
- `--error_col`: The name of the column containing model errors.
- `--error_type`: `binary` (e.g., 0=Correct, 1=Incorrect) or `regression` (continuous error values).
- *(For regression, you can optionally pass `--y_true_col` and `--y_pred_col` to have C4F compute the signed error automatically).*

### K-Selection & Scoring
- `--n_clusters`: Fixed number of clusters.
- `--n_min` / `--n_max`: Range to search for optimal `k`.
- `--scoring`: Metric to optimize during search:
  - `composite` (default): Weighted combination of silhouette, error separation, and demographic separation.
  - `silhouette`: Pure clustering quality.
  - `chi2_error`: Maximizes the statistical difference in errors across clusters.
  - `chi2_sensitive`: Maximizes the statistical difference in sensitive attributes across clusters.
- `--composite_weights`: Define weights for composite scoring (e.g., `silhouette:0.3,error:0.5,fairness:0.2`).

### Filtering
- `--min_datapoints`: Minimum cluster size. (Mapped to HDBSCAN's native `min_cluster_size`, acts as a post-hoc filter for other algorithms, moving small clusters to noise).
- `--subset`: Filter the analysis to a specific confusion matrix subset (`TP`, `TN`, `FP`, `FN`, `TP_TN`, `FP_FN`). Requires `--y_true_col` and `--y_pred_col`.

---

## Citation

If you use this software in your research, please cite:

```bibtex
@mastersthesis{bpai_thesis_2026,
  author       = {[Author Name]},
  title        = {[Thesis Title]},
  school       = {[University Name]},
  year         = {2026},
  url          = {https://github.com/[username]/BPAI}
}
```

## License

This project is open-source. Please see the `LICENSE` file for details.