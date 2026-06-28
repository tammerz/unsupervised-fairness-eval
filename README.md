# Auditing Automated Valuation Model Fairness Using Socioeconomic and Geographic Clustering

[![Degree: BSc Artificial Intelligence](https://img.shields.io/badge/Degree-BSc%20Artificial%20Intelligence-blue.svg)](https://vu.nl)
[![University: Vrije Universiteit Amsterdam](https://img.shields.io/badge/University-VU%20Amsterdam-0077B3.svg)](https://vu.nl)
[![Framework: C4F](https://img.shields.io/badge/Framework-C4F-orange.svg)](https://github.com/emma-ba/Clustering_4_Fairness)

This repository contains the official code for a Bachelor's thesis from the Artificial Intelligence department at **Vrije Universiteit Amsterdam**.

---

## Overview

This project provides a reproducible pipeline to audit prediction errors from Automated Valuation Models (AVMs) in **Los Angeles County**, using a dataset of 38,878 transactions.

Real estate algorithms can inadvertently perpetuate bias through **proxy leakage**, where spatial features like `latitude` and `longitude` act as substitutes for sensitive demographic data. This happens even when such data is excluded to comply with fair-housing laws. This project adapts the **Clustering for Fairness (C4F)** framework, originally developed by Beauxis-Aussalet (2024), to a regression context. It analyzes transaction errors (`logerror`) alongside the **Area Deprivation Index (ADI)** to uncover localized algorithmic harm.


## Repository Structure

```text
📦 BPAI
├── 📂 c4f
│   # Core C4F framework files
├── 📂 clustering_results
│   # Output from clustering experiments
├── 📂 Data
│   # Raw data and geographic shapefiles
├── 📂 examples
│   # Example scripts and notebooks
├── 📂 sanitychecks
│   # Notebooks for analysis and visualization
├── 📄 main.py
├── 📄 setup.py
├── 📄 requirements.txt
└── 📄 README.md
```
---

## Key Contributions

*   **Framework Extension for Regression:** Adapts the C4F auditing framework from classification to regression, using Gower similarity and K-Medoids clustering to handle continuous data.
*   **Dimensional Parity:** Introduces a feature-weighting strategy to balance the influence of geographic coordinates against socioeconomic data, ensuring a more equitable analysis.
*   **Evidence of Systematic Bias:** Provides empirical evidence of a **U-shaped bias curve**, revealing that models tend to overvalue properties in deprived areas and undervalue them in affluent ones.

---

## Experimental Configurations

The pipeline evaluates three core conditions to distinguish the effects of spatial proximity from socioeconomic factors:

| Experiment Tag | Feature Space Configuration | ADI Weight ($\alpha$) | Clusters ($k$) | Auditing Objective |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | `+REG+SEN+ERR` | `1.0` | `5` | Discovers broad regions and confirms the dominance of geographic features. |
| **Optimised Geographic** | `+REG+SEN+ERR` | `2.0` | `28` | Balances feature influence to uncover localized, directional harm. |
| **Socioeconomic Isolation**| `-reg+SEN+ERR` | `1.0` | `10` | Isolates socioeconomic factors to confirm the U-shaped error pattern. |

---

## Quickstart

**1. Set up the environment**
```bash
git clone https://github.com/your-username/BPAI.git
cd BPAI
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**2. Run the Zillow analysis**
```bash
cd examples/zillow_case_study
run_zillow_analysis.bat
```
You can modify clustering seeds and feature weights by editing the `run_zillow_analysis.bat` script.

---

## Citation

This work is an adaptation of the Clustering for Fairness (C4F) repository.

```bibtex
@software{beauxisaussalet_c4f,
  author       = {Beauxis-Aussalet, Emmanuelle},
  title        = {Clustering\_4\_Fairness},
  url          = {https://github.com/emma-ba/Clustering_4_Fairness},
  version      = {b1c712fdba3791e5fe8da4b45c0b15507de7fc1c},
  date         = {2024},
}
```