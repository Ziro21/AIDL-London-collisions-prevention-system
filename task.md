# Phase 1: Project Setup Tasks (Completed)

- [x] Create git branch `draft1`
- [x] Scaffold project directories (`data`, `notebooks`, `outputs`)
- [x] Create foundation files (`README.md`, `.gitignore`, `requirements.txt`)
- [x] Initialize empty Jupyter notebooks
- [x] Cleanup extraneous `.claude` worktree folders

# Phase 2: Data Acquisition & EDA

- [x] Download STATS19 2024 CSVs into `data/` folder
- [x] Initialize `01_data_loading_and_eda.ipynb` with seeds and imports
- [x] Implement data joining strategy (Collisions + Worst Casualty + First Vehicle)
- [x] Filter dataset to exact Greater London Boundaries (ONS `E09` code)
- [x] Perform Advanced Exploratory Data Analysis (EDA) and generate elite visualisations

# Phase 3: Advanced Preprocessing (`02_preprocessing.ipynb`)

- [x] Select strict prediction-time feature sets
- [x] Remap Target Variable to start at 0 for PyTorch (Fatal=0, Serious=1, Slight=2)
- [x] Advanced Imputation (KNN Imputer for numeric/categorical missing values)
- [x] Target Encoding (Target/WoE encoding for high-cardinality nominal variables)
- [x] Implement Stratified Train/Val/Test Split (70% / 15% / 15%)
- [x] Compute Class Weights for Focal Loss (implemented in Notebook 4 training loop)
- [x] Standard Scale Features and securely save the Scaler for inference

# Phase 4: Baseline Model (`03_baseline_model.ipynb`)

- [x] Load preprocessed artefacts from `preprocessed_data.joblib`
- [x] Train Random Forest with class weights (scikit-learn)
- [x] Evaluate: Macro F1, per-class Precision/Recall, Confusion Matrix
- [x] Document baseline performance as the benchmark to beat

# Phase 5: MLP Deep Learning Model (`04_mlp_model.ipynb`)

- [x] Build PyTorch `Dataset` and `DataLoader` classes
- [x] Design MLP architecture (BatchNorm, Dropout, ReLU)
- [ ] Implement custom Focal Loss with class weights
- [ ] Training loop with early stopping and LR scheduling
- [ ] Hyperparameter tuning (hidden dims, dropout, LR, focal gamma)
- [ ] Save best model checkpoint to `outputs/models/`

# Phase 6: Evaluation & Comparison (`05_evaluation.ipynb`)

- [ ] Load Baseline + MLP models and generate predictions
- [ ] Confusion matrices (normalised, per-class)
- [ ] ROC curves and AUC-ROC (one-vs-rest, per-class)
- [ ] Precision-Recall curves (critical for Fatal class)
- [ ] Comprehensive metrics table: Accuracy, Macro F1, Weighted F1, Fatal Recall
- [ ] Statistical significance testing (McNemar's test)
- [ ] Professional visualisations for presentation

# Phase 7: Interpretability & Ethics (`06_interpretability.ipynb`)

- [ ] SHAP analysis: global feature importance (beeswarm + bar plots)
- [ ] SHAP analysis: local explanations for individual Fatal predictions
- [ ] LIME explanations for misclassified Fatal crashes
- [ ] Bias audit: model fairness across London boroughs and time periods
- [ ] Ethics discussion: deployment risks, false negative consequences, data privacy
- [ ] Final recommendations and limitations report for IntelliSys
