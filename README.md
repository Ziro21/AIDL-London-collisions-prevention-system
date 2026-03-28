# AIDL London Collisions Prevention System

## Project Overview
This project is developed for **IntelliSys Ltd.** to demonstrate the application of deep learning in predicting road collision injury severity (slight, serious, fatal) based on environmental and contextual conditions present at the time of a crash occurring. The scope targets Greater London using the UK Department for Transport's STATS19 Road Safety Open Data (2024).

By predicting collision severity, IntelliSys can shift transport clients from a reactive emergency response to predictive risk management.

## Repository Structure
- `data/`: Raw CSV files from STATS19 (ignored in version control).
- `notebooks/`: Jupyter notebooks covering the full pipeline from data loading to model evaluation and ethical considerations.
- `outputs/`: Model weights, generated figures, and interpretability visualisations (SHAP).

## Setup Instructions
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the STATS19 2024 data and place the CSV files in `data/`.
4. Run the notebooks in sequence.
