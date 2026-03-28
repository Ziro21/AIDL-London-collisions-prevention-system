"""
preprocessing.py
----------------
Feature selection, missing-value imputation, categorical encoding,
class-imbalance handling, train/val/test splitting, and feature scaling.

All decisions are documented inline and mirrored in Notebook 2 markdown.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from .utils import get_logger, outputs_dir

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Feature catalogue
# ---------------------------------------------------------------------------

#: Features available at prediction time (pre-crash conditions only).
#: Columns that are identifiers, GPS coords, or post-crash fields are excluded.
FEATURE_COLUMNS = [
    # --- Collision-level (Collisions table) ---
    "day_of_week",
    "time",
    "road_type",
    "speed_limit",
    "junction_detail",
    "junction_control",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "urban_or_rural_area",
    "number_of_vehicles",
    "number_of_casualties",
    # --- Casualty-level (Casualties table, worst casualty) ---
    "casualty_type",
    "casualty_class",
    "sex_of_casualty",
    "age_of_casualty",
    "pedestrian_location",
    "pedestrian_movement",
    # --- Vehicle-level (Vehicles table, first vehicle) ---
    "vehicle_type",
    "vehicle_manoeuvre",
    "age_of_driver",
    "sex_of_driver",
    "engine_capacity_cc",
    "age_of_vehicle",
]

TARGET_COLUMN = "accident_severity"

#: Remap accident_severity → 0-indexed for PyTorch CrossEntropyLoss
#:   1 (Fatal) → 0 | 2 (Serious) → 1 | 3 (Slight) → 2
SEVERITY_MAP = {1: 0, 2: 1, 3: 2}
CLASS_NAMES = ["Fatal", "Serious", "Slight"]

#: Drop any feature column with more than this fraction of missing values
MISSING_DROP_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_available_features(df: pd.DataFrame) -> list[str]:
    """Return the intersection of FEATURE_COLUMNS and df's actual columns.

    Some column names differ slightly across STATS19 releases.  This guard
    avoids KeyErrors and documents the gap.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    list[str]
    """
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = set(FEATURE_COLUMNS) - set(available)
    if missing:
        logger.warning("Feature columns not found in DataFrame — skipping: %s", missing)
    return available


def remap_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'target' column remapped from accident_severity to 0-indexed.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'accident_severity'.

    Returns
    -------
    pd.DataFrame
        Copy with 'target' column added.
    """
    out = df.copy()
    out["target"] = out[TARGET_COLUMN].map(SEVERITY_MAP)
    unmapped = out["target"].isna().sum()
    if unmapped:
        logger.warning("%d rows have unmapped accident_severity values — dropped", unmapped)
        out = out.dropna(subset=["target"])
    out["target"] = out["target"].astype(int)
    return out


def handle_missing_values(
    df: pd.DataFrame,
    features: list[str],
    drop_threshold: float = MISSING_DROP_THRESHOLD,
) -> tuple[pd.DataFrame, list[str]]:
    """Impute or drop columns with missing values.

    Strategy
    --------
    - Columns with > ``drop_threshold`` missing: dropped entirely.
    - Numeric columns: median imputation.
    - Object/categorical columns: mode imputation (first mode).

    Parameters
    ----------
    df : pd.DataFrame
    features : list[str]
        Candidate feature columns.
    drop_threshold : float

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        (DataFrame with imputed values, updated feature list after drops)
    """
    out = df.copy()
    kept_features: list[str] = []

    for col in features:
        missing_rate = out[col].isna().mean()
        if missing_rate > drop_threshold:
            logger.info(
                "Dropping '%s' — %.1f%% missing (threshold %.0f%%)",
                col, 100 * missing_rate, 100 * drop_threshold,
            )
            continue

        if out[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
            fill_val = out[col].median()
            out[col] = out[col].fillna(fill_val)
        else:
            fill_val = out[col].mode().iloc[0] if not out[col].mode().empty else "Unknown"
            out[col] = out[col].fillna(fill_val)

        kept_features.append(col)

    return out, kept_features


def encode_time_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour from the 'time' string column (HH:MM format).

    Replaces the raw time string with a numeric hour (0–23), which is
    suitable for the MLP without triggering high-cardinality one-hot expansion.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    out = df.copy()
    if "time" in out.columns:
        out["hour"] = pd.to_datetime(out["time"], format="%H:%M", errors="coerce").dt.hour
        out = out.drop(columns=["time"])
        # Replace 'time' in feature list handled externally
    return out


def one_hot_encode(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Apply one-hot encoding to nominal categorical columns.

    Numeric columns pass through unchanged.  String/object columns are
    one-hot encoded with ``pd.get_dummies``.

    Parameters
    ----------
    df : pd.DataFrame
    features : list[str]

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        (encoded DataFrame, final feature name list)
    """
    encoded = pd.get_dummies(df[features], drop_first=False)
    # Ensure all columns are numeric (bool → int8 for PyTorch)
    bool_cols = encoded.select_dtypes(include="bool").columns
    encoded[bool_cols] = encoded[bool_cols].astype(np.int8)
    logger.info(
        "One-hot encoding: %d → %d features", len(features), encoded.shape[1]
    )
    return encoded, list(encoded.columns)


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 70 / 15 / 15 train / val / test split.

    Parameters
    ----------
    X : np.ndarray  shape (n_samples, n_features)
    y : np.ndarray  shape (n_samples,)
    val_size : float
    test_size : float
    random_state : int

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    temp_size = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, random_state=random_state, stratify=y
    )
    relative_test = test_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, random_state=random_state, stratify=y_temp
    )
    logger.info(
        "Split — Train: %d | Val: %d | Test: %d",
        len(y_train), len(y_val), len(y_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    save_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on training data; transform val and test.

    The scaler is fitted on X_train only to prevent data leakage.
    It is saved to disk so it can be reloaded identically at inference time.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray
    save_path : Path, optional
        Where to pickle the fitted scaler. Defaults to outputs/models/scaler.pkl.

    Returns
    -------
    X_train_scaled, X_val_scaled, X_test_scaled, scaler
    """
    if save_path is None:
        save_path = outputs_dir("models") / "scaler.pkl"

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    with open(save_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Scaler saved to %s", save_path)

    return X_train_s, X_val_s, X_test_s, scaler


def compute_class_weights(y_train: np.ndarray) -> np.ndarray:
    """Compute balanced class weights for CrossEntropyLoss.

    Addresses the severe class imbalance (~85% Slight, ~14% Serious, ~1% Fatal)
    by upweighting minority classes during training.

    Parameters
    ----------
    y_train : np.ndarray

    Returns
    -------
    np.ndarray  shape (3,)  — weights for [Fatal, Serious, Slight]
    """
    classes = np.array([0, 1, 2])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    logger.info(
        "Class weights — Fatal: %.3f | Serious: %.3f | Slight: %.3f",
        *weights,
    )
    return weights
