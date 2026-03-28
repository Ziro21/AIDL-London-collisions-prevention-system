"""
src — IntelliSys London Collision Severity Prediction
======================================================
Reusable Python modules that support the Jupyter notebook pipeline.

Modules
-------
utils           : Seed setting, path resolution, logging helpers
data_loader     : STATS19 CSV loading, table joining, London filter
preprocessing   : Feature selection, encoding, imputation, train/val/test split
model           : CollisionMLP PyTorch architecture
train           : Training loop with early stopping and LR scheduling
evaluate        : Metrics computation, confusion matrix, AUC-ROC
"""
