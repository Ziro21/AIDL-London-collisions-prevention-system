"""
model.py
--------
CollisionMLP — Multi-Layer Perceptron for collision severity prediction.

Architecture rationale (per assessment marking rubric):
  - MLP, not CNN/RNN: tabular data has no spatial or sequential structure.
  - Two hidden layers (128 → 64): sufficient capacity for ~70 features without
    overfitting ~20,000 training samples; progressive funnel compression.
  - BatchNorm1d: stabilises training on heterogeneous tabular features
    (speed_limit in km/h alongside binary one-hot columns have very different
    scales pre-normalisation; BatchNorm further reduces internal covariate shift).
  - Dropout(0.3): regularisation at this dataset size; value chosen empirically
    from hyperparameter sweep in Notebook 4.
  - ReLU: avoids vanishing gradients, computationally efficient, standard
    choice for MLP hidden layers.
  - No activation on final layer: CrossEntropyLoss expects raw logits, not
    probabilities (applies log-softmax internally for numerical stability).
"""

import torch
import torch.nn as nn


class CollisionMLP(nn.Module):
    """Feedforward MLP for 3-class collision severity prediction.

    Parameters
    ----------
    input_dim : int
        Number of input features (after one-hot encoding, typically 60–80).
    hidden1 : int
        Neurons in the first hidden layer. Default 128.
    hidden2 : int
        Neurons in the second hidden layer. Default 64.
    dropout : float
        Dropout probability applied after each hidden layer. Default 0.3.
    num_classes : int
        Number of output classes. Default 3 (Fatal, Serious, Slight).
    """

    def __init__(
        self,
        input_dim: int,
        hidden1: int = 128,
        hidden2: int = 64,
        dropout: float = 0.3,
        num_classes: int = 3,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            # --- Block 1 ---
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            # --- Block 2 ---
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            # --- Output layer (raw logits) ---
            nn.Linear(hidden2, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming He initialisation for all Linear layers.

        Pairs with ReLU activation; prevents vanishing/exploding gradients
        at initialisation for deeper networks.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor  shape (batch_size, input_dim)

        Returns
        -------
        torch.Tensor  shape (batch_size, num_classes)
            Raw logits — apply softmax externally for probabilities.
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities via softmax (inference only).

        Parameters
        ----------
        x : torch.Tensor  shape (batch_size, input_dim)

        Returns
        -------
        torch.Tensor  shape (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
