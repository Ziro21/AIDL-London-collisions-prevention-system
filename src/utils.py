"""
utils.py
--------
Shared utilities: reproducibility seeds, path management, logging.

All notebooks import set_seeds() at their top cell to guarantee
identical results across runs (required by assessment rubric).
"""

import os
import random
import logging
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility.

    Parameters
    ----------
    seed : int
        Seed value used for numpy and Python's random module.
        Pass torch.manual_seed separately in notebooks that import torch,
        because torch is an optional dependency here.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def project_root() -> Path:
    """Return the absolute path to the repository root.

    Resolves from this file's location so notebooks can use relative imports
    regardless of where they are launched from.

    Returns
    -------
    Path
        Absolute path to the project root directory.
    """
    return Path(__file__).resolve().parent.parent


def data_dir(subfolder: str = "raw") -> Path:
    """Return path to a data subdirectory, creating it if absent.

    Parameters
    ----------
    subfolder : str
        One of ``'raw'`` or ``'processed'``.

    Returns
    -------
    Path
        Absolute path to ``<root>/data/<subfolder>/``.
    """
    path = project_root() / "data" / subfolder
    path.mkdir(parents=True, exist_ok=True)
    return path


def outputs_dir(subfolder: str = "figures") -> Path:
    """Return path to an outputs subdirectory, creating it if absent.

    Parameters
    ----------
    subfolder : str
        One of ``'figures'``, ``'models'``, or ``'shap'``.

    Returns
    -------
    Path
        Absolute path to ``<root>/outputs/<subfolder>/``.
    """
    path = project_root() / "outputs" / subfolder
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger.

    Parameters
    ----------
    name : str
        Logger name — typically ``__name__`` of the calling module.
    level : int
        Logging level (default INFO).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
