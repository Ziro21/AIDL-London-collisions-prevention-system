"""
train.py
--------
Training loop for CollisionMLP with early stopping and LR scheduling.

Design decisions:
  - Adam optimiser (lr=1e-3 default): adaptive learning rate, robust default
    for tabular MLP problems.
  - ReduceLROnPlateau scheduler: halves LR when validation loss plateaus for
    `scheduler_patience` epochs; avoids manual LR tuning.
  - Early stopping (patience=10): prevents overfitting on ~20k samples;
    restores best weights on exit.
  - Weighted CrossEntropyLoss: addresses the ~85% Slight class imbalance by
    upweighting Fatal and Serious classes during gradient updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .utils import get_logger, outputs_dir

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters and training settings.

    Attributes
    ----------
    epochs : int        Maximum training epochs.
    batch_size : int    Mini-batch size for SGD.
    lr : float          Initial learning rate for Adam.
    patience : int      Early stopping patience (epochs without val improvement).
    scheduler_patience : int  LR scheduler patience.
    scheduler_factor : float  LR reduction factor on plateau.
    model_save_path : Path    Where to save the best model checkpoint.
    """

    epochs: int = 150
    batch_size: int = 64
    lr: float = 1e-3
    patience: int = 10
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    model_save_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.model_save_path is None:
            self.model_save_path = outputs_dir("models") / "best_model.pt"


@dataclass
class TrainingHistory:
    """Records per-epoch loss for plotting learning curves.

    Attributes
    ----------
    train_losses : list[float]
    val_losses   : list[float]
    best_epoch   : int          Epoch at which the best val loss was achieved.
    """

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_epoch: int = 0


def build_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Wrap numpy arrays in a PyTorch DataLoader.

    Parameters
    ----------
    X : np.ndarray  float32
    y : np.ndarray  int64
    batch_size : int
    shuffle : bool

    Returns
    -------
    DataLoader
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: np.ndarray,
    config: Optional[TrainingConfig] = None,
) -> TrainingHistory:
    """Train CollisionMLP with early stopping.

    Parameters
    ----------
    model : nn.Module
        Instantiated CollisionMLP.
    X_train, y_train : np.ndarray
        Training features and labels.
    X_val, y_val : np.ndarray
        Validation features and labels.
    class_weights : np.ndarray  shape (3,)
        Balanced class weights from ``preprocessing.compute_class_weights``.
    config : TrainingConfig, optional
        Training hyperparameters. Uses defaults if None.

    Returns
    -------
    TrainingHistory
        Loss curves and best epoch index.
    """
    if config is None:
        config = TrainingConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info("Training on device: %s", device)

    train_loader = build_dataloader(X_train, y_train, config.batch_size, shuffle=True)

    weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_t)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=config.scheduler_patience, factor=config.scheduler_factor
    )

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t  = torch.tensor(y_val, dtype=torch.long).to(device)

    history = TrainingHistory()
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.epochs):
        # --- Training phase ---
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # --- Validation phase ---
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        history.train_losses.append(avg_train_loss)
        history.val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history.best_epoch = epoch
            torch.save(model.state_dict(), config.model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter == config.patience:
            logger.info(
                "Epoch %3d | train_loss: %.4f | val_loss: %.4f | patience: %d/%d",
                epoch, avg_train_loss, val_loss, patience_counter, config.patience,
            )

        if patience_counter >= config.patience:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    # Restore best weights
    model.load_state_dict(torch.load(config.model_save_path, map_location=device))
    logger.info(
        "Training complete. Best val_loss %.4f at epoch %d",
        best_val_loss, history.best_epoch,
    )
    return history
