# AIDL London Collisions Prevention System

## Project Overview
Developed for **IntelliSys Ltd.** to predict road collision injury severity (Fatal, Serious, Slight) in Greater London using the UK Department for Transport's STATS19 Road Safety Open Data (2024).

By predicting collision severity from environmental and contextual conditions, IntelliSys can shift transport clients from reactive emergency response to **predictive risk management**.

## Key Results

| Model | Fatal Recall | Macro F1 | MCC | ROC-AUC (Fatal) |
|-------|-------------|----------|-----|-----------------|
| RF Baseline | 0.0% | 0.309 | 0.028 | — |
| **MLP (argmax)** | **43.8%** | **0.326** | **0.100** | **0.738** |
| MLP (τ=0.13) | 68.8% | 0.276 | 0.068 | 0.738 |

The MLP detects fatal crashes that the Random Forest completely misses, with a Fatal ROC-AUC of 0.738 confirming strong discriminative ability.

## Pipeline

| Notebook | Phase | Description |
|----------|-------|-------------|
| `01_data_loading_and_eda.ipynb` | Data & EDA | STATS19 loading, London filtering, advanced visualisations |
| `02_preprocessing.ipynb` | Preprocessing | Target encoding, KNN imputation, stratified splits, scaling |
| `03_baseline_model.ipynb` | Baseline | Random Forest (300 trees, balanced weights) |
| `04_mlp_model.ipynb` | Deep Learning | MLP with Focal Loss, hyperparameter sweep, ensemble, K-fold CV |
| `05_evaluation.ipynb` | Evaluation | Test-set results, ROC/PR curves, McNemar's test |
| `05_hierarchical_pipeline.ipynb` | Experimental | Two-stage Fatal/Non-Fatal pipeline with SMOTE |
| `06_interpretability_and_ethics.ipynb` | Interpretability | SHAP, LIME, bias audit, ethics, recommendations |

## Top Predictive Features (SHAP)
1. **Road type** — strongest predictor across all severity classes
2. **Pedestrian crossing** — infrastructure-level risk factor
3. **Light conditions** — environmental driver
4. **Number of vehicles** — collision complexity
5. **Max driver age** — human factor

## Repository Structure
- `data/` — Raw STATS19 CSVs (gitignored)
- `notebooks/` — Full Jupyter pipeline (7 notebooks)
- `outputs/figures/` — EDA plots, training curves, ROC/PR curves, model comparison
- `outputs/models/` — Model checkpoints, preprocessed data, evaluation artefacts
- `outputs/shap/` — SHAP/LIME visualisations and saved values

## Setup
```bash
git clone https://github.com/Ziro21/AIDL-London-collisions-prevention-system.git
cd AIDL-London-collisions-prevention-system
pip install -r requirements.txt
```
Download the [STATS19 2024 data](https://data.dft.gov.uk/road-accidents-safety-data/) and place CSV files in `data/`. Run notebooks in sequence.

## Tech Stack
Python 3.12 · PyTorch · scikit-learn · SHAP · LIME · pandas · matplotlib
