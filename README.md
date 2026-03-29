# AIDL London Collisions Prevention System

**Client:** IntelliSys Ltd.
**Module:** WM9B7-15 Artificial Intelligence & Deep Learning
**University:** WMG, University of Warwick — MSc Applied AI 2025/26
**Presentation date:** Week commencing 20 April 2026

---

## Problem

Predict road collision injury severity (Fatal / Serious / Slight) from conditions
present at the time of a crash using UK STATS19 2024 data, filtered to Greater London.
Enables IntelliSys to shift transport clients from reactive emergency response to
**predictive risk management**.

---

## Repository Structure

```
AIDL-London-collisions-prevention-system/
│
├── data/
│   ├── raw/                  ← STATS19 2024 CSVs (gitignored — download separately)
│   └── processed/            ← merged arrays & feature names (gitignored)
│
├── notebooks/
│   ├── 01_data_loading_and_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_mlp_model.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_interpretability_and_ethics.ipynb
│
├── src/
│   ├── __init__.py
│   ├── utils.py              ← seeds, path helpers, logging
│   ├── data_loader.py        ← STATS19 loading, joining, London filter
│   ├── preprocessing.py      ← feature selection, encoding, split, scaling
│   ├── model.py              ← CollisionMLP architecture
│   ├── train.py              ← training loop, early stopping
│   └── evaluate.py           ← metrics, confusion matrix, AUC-ROC
│
├── config/
│   └── config.yaml           ← hyperparameters, paths, deployment thresholds
│
├── outputs/
│   ├── figures/              ← saved plots (.gitignored except .gitkeep)
│   ├── models/               ← model weights + scaler (.gitignored except .gitkeep)
│   └── shap/                 ← SHAP and LIME outputs
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone
git clone <repo-url>
cd AIDL-London-collisions-prevention-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download STATS19 2024 data
#    https://www.gov.uk/government/statistical-data-sets/road-safety-open-data
#    Place the three CSVs in data/raw/

# 4. Run notebooks in order (01 → 06)
jupyter notebook
```

---

## Dataset

| File | Description |
|------|-------------|
| `dft-road-casualty-statistics-collision-2024.csv` | One row per collision (master table) |
| `dft-road-casualty-statistics-casualty-2024.csv`  | One row per casualty |
| `dft-road-casualty-statistics-vehicle-2024.csv`   | One row per vehicle |

**London filter:** `police_force ∈ {1, 48}` (Metropolitan + City of London Police)
**Target:** `accident_severity` — 1=Fatal, 2=Serious, 3=Slight (remapped to 0/1/2)
**Licence:** Open Government Licence v3.0

---

## Model

`CollisionMLP` — two-layer feedforward MLP (128 → 64 → 3):

- **BatchNorm + Dropout(0.3)** per hidden layer
- **Weighted CrossEntropyLoss** to handle ~85% Slight / ~14% Serious / ~1% Fatal imbalance
- **Adam** optimiser with `ReduceLROnPlateau` and early stopping

---

## Key Evaluation Metric

**Fatal recall** — the most safety-critical metric. A false negative on Fatal
(predicting 'Slight' when the collision is actually fatal) prevents ambulance
pre-alerting in IntelliSys's deployment scenario.
