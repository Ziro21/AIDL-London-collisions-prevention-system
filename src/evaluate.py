"""
evaluate.py
-----------
Evaluation helpers: metrics, confusion matrix, ROC-AUC, and comparison tables.

The functions here are used in Notebook 5.  Every metric is computed on
the held-out test set — never on training or validation data.

Clinical interpretation note
-----------------------------
Fatal recall is the most safety-critical single metric in this domain.
A false negative on Fatal means the model predicted "Slight" when the
collision was actually fatal — in deployment, this prevents the ambulance
station from being pre-alerted.  Evaluation commentary must address this.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from .utils import get_logger, outputs_dir

logger = get_logger(__name__)

CLASS_NAMES = ["Fatal", "Serious", "Slight"]


def predict(
    model: nn.Module,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return predicted labels and class probabilities.

    Parameters
    ----------
    model : nn.Module  CollisionMLP (weights already loaded)
    X : np.ndarray     shape (n_samples, n_features)  float32

    Returns
    -------
    y_pred : np.ndarray  shape (n_samples,)  int  class indices
    probs  : np.ndarray  shape (n_samples, 3) float softmax probabilities
    """
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_t)
        probs  = torch.softmax(logits, dim=1).numpy()
    y_pred = np.argmax(probs, axis=1)
    return y_pred, probs


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print sklearn classification report with domain-appropriate class names.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    """
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save: bool = True,
) -> np.ndarray:
    """Plot and optionally save the normalised confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    save : bool  If True, write PNG to outputs/figures/.

    Returns
    -------
    np.ndarray  Raw (unnormalised) confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        path = outputs_dir("figures") / "confusion_matrix.png"
        fig.savefig(path, dpi=150)
        logger.info("Confusion matrix saved to %s", path)

    plt.show()
    return cm


def compute_auc_roc(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Compute macro-averaged one-vs-rest AUC-ROC.

    Parameters
    ----------
    y_true : np.ndarray  int class labels
    probs  : np.ndarray  shape (n, 3) softmax probabilities

    Returns
    -------
    float  Macro AUC-ROC (higher = better; 1.0 = perfect)
    """
    auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    logger.info("Macro AUC-ROC: %.4f", auc)
    return auc


def build_comparison_table(
    y_test: np.ndarray,
    y_pred_lr: np.ndarray,
    y_pred_mlp: np.ndarray,
    probs_lr: np.ndarray,
    probs_mlp: np.ndarray,
) -> pd.DataFrame:
    """Build a side-by-side metrics comparison table (Logistic Reg vs MLP).

    Parameters
    ----------
    y_test              : true test labels
    y_pred_lr, y_pred_mlp : predictions from each model
    probs_lr, probs_mlp   : probability arrays from each model

    Returns
    -------
    pd.DataFrame  with rows = metric names, cols = ['Logistic Regression', 'MLP']
    """
    from sklearn.metrics import accuracy_score, f1_score

    def _metrics(y_true, y_pred, probs):
        report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
        return {
            "Accuracy":        round(accuracy_score(y_true, y_pred), 4),
            "Macro F1":        round(report["macro avg"]["f1-score"], 4),
            "Weighted F1":     round(report["weighted avg"]["f1-score"], 4),
            "Fatal Recall":    round(report["Fatal"]["recall"], 4),
            "Serious Recall":  round(report["Serious"]["recall"], 4),
            "Macro AUC-ROC":   round(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"), 4),
        }

    lr_metrics  = _metrics(y_test, y_pred_lr,  probs_lr)
    mlp_metrics = _metrics(y_test, y_pred_mlp, probs_mlp)

    table = pd.DataFrame(
        {"Logistic Regression": lr_metrics, "MLP": mlp_metrics}
    )
    return table


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    save: bool = True,
) -> None:
    """Plot training and validation loss curves.

    Parameters
    ----------
    train_losses : list[float]
    val_losses   : list[float]
    save : bool
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label="Train loss", linewidth=1.5)
    ax.plot(val_losses,   label="Val loss",   linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss Curves", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()

    if save:
        path = outputs_dir("figures") / "training_curves.png"
        fig.savefig(path, dpi=150)
        logger.info("Training curves saved to %s", path)

    plt.show()
