# Auditing Automated Valuation Model Fairness Using Socioeconomic and Geographic Clustering

[![Degree: BSc Artificial Intelligence](https://img.shields.io/badge/Degree-BSc%20Artificial%20Intelligence-blue.svg)](https://vu.nl)
[![University: Vrije Universiteit Amsterdam](https://img.shields.io/badge/University-VU%20Amsterdam-0077B3.svg)](https://vu.nl)
[![Framework: C4F](https://img.shields.io/badge/Framework-C4F-orange.svg)](https://github.com/emma-ba/Clustering_4_Fairness)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the official code for a Bachelor's thesis from the Artificial Intelligence department at **Vrije Universiteit Amsterdam**.

**Author:** T. Muñoz Jordá

---

## Overview

This project provides a reproducible pipeline to audit prediction errors from Automated Valuation Models (AVMs) in **Los Angeles County**, using a dataset of 38,878 transactions.

Real estate algorithms can inadvertently perpetuate bias through **proxy leakage**, where spatial features like `latitude` and `longitude` act as substitutes for sensitive demographic data. This happens even when such data is excluded to comply with fair-housing laws. This project adapts the **Clustering for Fairness (C4F)** framework, originally developed by Beauxis-Aussalet (2024), to a regression context. It analyzes transaction errors (`logerror`) alongside the **Area Deprivation Index (ADI)** to uncover localized algorithmic harm.

---

## Getting Started: Installation and Usage

This guide provides a complete workflow for setting up the environment, preparing the data, and running the experiments.

### 1. Set up the Environment
```bash
# Clone the repository
git clone https://github.com/tammerz/unsupervised-fairness-eval.git
cd unsupervised-fairness-eval

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Download the Data

To run this analysis, you will need to download three external datasets and place them in the `Data/` directory:

1.  **Zillow Valuation Data (2016)**
    *   **Where to go:** [Kaggle Zillow Prize Competition](https://www.kaggle.com/c/zillow-prize-1)
    *   **Files to download:** `train_2016.csv` and `properties_2016.csv`
    *   *Note:* You must sign into a Kaggle account and accept the competition rules to access the files.

2.  **Area Deprivation Index (2015)**
    *   **Where to go:** [The Neighborhood Atlas](https://www.neighborhoodatlas.medicine.wisc.edu/)
    *   **Files to download:** 2015 ADI dataset for **California**
    *   *Note:* Requires a free account. Ensure your selected download includes the 12-digit Census FIPS identifiers.

3.  **Census Block Group Boundaries (2015)**
    *   **Where to go:** [U.S. Census Bureau TIGER/Line Shapefiles](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.2015.html)
    *   **Files to download:** The 2015 **Block Groups** shapefile layer for **California**

### 3. Run the Zillow Experiments

 The command below executes the "Optimised Geographic" experiment, which is the main configuration discussed in the thesis.

```bash
cd examples/zillow_case_study
run_zillow_analysis.bat
```
You can modify clustering seeds and feature weights by editing the `run_zillow_analysis.bat` script.

---

## Experimental Configurations

The pipeline evaluates three core conditions to distinguish the effects of spatial proximity from socioeconomic factors. You can run them using the CLI as shown in the usage section.

| Experiment Tag          | Feature Space Configuration | ADI Weight ($\alpha$) | Clusters ($k$) | Auditing Objective                                                    |
| :---------------------- | :-------------------------- | :-------------------- | :------------- | :-------------------------------------------------------------------- |
| **Baseline**            | `+REG+SEN+ERR`              | `1.0`                 | `5`            | Discovers broad regions and confirms the dominance of geographic features. |
| **Optimised Geographic**| `+REG+SEN+ERR`              | `2.0`                 | `28`           | Balances feature influence to uncover localized, directional harm.    |
| **Socioeconomic**       | `-reg+SEN+ERR`              | `1.0`                 | `10`           | Isolates socioeconomic factors to confirm the U-shaped error pattern. |

---

## Key Contributions

*   **Framework Extension for Regression:** Adapts the C4F auditing framework from classification to regression, using Gower similarity and K-Medoids clustering to handle continuous data.
*   **Dimensional Parity:** Introduces a feature-weighting strategy to balance the influence of geographic coordinates against socioeconomic data, ensuring a more equitable analysis.
*   **Evidence of Systematic Bias:** Provides empirical evidence of a **U-shaped bias curve**, revealing that models tend to overvalue properties in deprived areas and undervalue them in affluent ones.

---

## Repository Structure

```text
unsupervised-fairness-eval/
├── main.py                       # CLI entry point for experiments
├── setup.py                      # Package definition for pip install -e .
├── requirements.txt
│
├── c4f/                          # Core library
│   ├── clustering.py             # K-Medoids clustering, Gower distance
│   ├── scoring.py                # Silhouette, Chi-2, Kruskal-Wallis scorers
│   ├── experiments.py            # Experiment orchestration and result aggregation
│   ├── visualization.py          # Scatter plots, composition bars, heatmaps
│   ├── fairness_metrics.py       # Demographic parity, representation ratio
│   └── preprocessing.py          # Categorical encoding
│
├── examples/zillow_case_study/   # Scripts for the Zillow AVM case study
│   ├── run_zillow_analysis.bat   # Batch script to run the full analysis pipeline
│   ├── prepare_zillow_input_data.py # Prepares Zillow data for analysis
│   └── spatial_join.py           # Joins transaction data with geographic shapefiles
│
├── sanitychecks/                 # Jupyter notebooks for validation and analysis
│   └── thesis_analysis.ipynb     # Primary notebook for thesis results and figures
│
└── Data/                         # Datasets and shapefiles (not versioned)
```

---

## Citation

If you use this work, please cite the original C4F framework and this thesis.

**This Thesis**
```bibtex
@misc{munoz_jorda_2024_avm,
  author       = {T. Muñoz Jordá},
  title        = {Auditing Automated Valuation Model Fairness Using Socioeconomic and Geographic Clustering},
  year         = {2024},
  publisher    = {Vrije Universiteit Amsterdam},
  url          = {https://github.com/tammerz/unsupervised-fairness-eval}
}
```

**Original C4F Framework**
```bibtex
@software{beauxisaussalet_c4f,
  author       = {Beauxis-Aussalet, Emmanuelle},
  title        = {Clustering\_4\_Fairness},
  url          = {https://github.com/emma-ba/Clustering_4_Fairness},
  version      = {b1c712fdba3791e5fe8da4b45c0b15507de7fc1c},
  date         = {2024},
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```