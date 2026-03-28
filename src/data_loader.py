"""
data_loader.py
--------------
Loads the three STATS19 2024 CSV files, joins them into a single
collision-level DataFrame, and filters to Greater London.

Join strategy (documented per assessment requirements):
  - Casualties : keep the most severely injured casualty per collision
    (sort casualty_severity ascending — 1=Fatal is worst — then groupby first).
  - Vehicles   : keep the first reported vehicle per collision
    (groupby accident_index, first row). Limitation: multi-vehicle crashes
    are simplified to the primary vehicle only.
  - Join order : collisions → left merge casualties → left merge vehicles.
  - London filter: police_force.isin([1, 48])
    (1 = Metropolitan Police, 48 = City of London Police).
"""

from pathlib import Path

import pandas as pd

from .utils import get_logger, data_dir

logger = get_logger(__name__)

# STATS19 2024 filenames (UK DfT open data)
_FILENAMES = {
    "collisions": "dft-road-casualty-statistics-collision-2024.csv",
    "casualties": "dft-road-casualty-statistics-casualty-2024.csv",
    "vehicles":   "dft-road-casualty-statistics-vehicle-2024.csv",
}

# Metropolitan Police = 1, City of London Police = 48
_LONDON_FORCES = [1, 48]


def load_raw_tables(raw_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three raw STATS19 CSVs from disk.

    Parameters
    ----------
    raw_dir : Path, optional
        Directory containing the CSV files. Defaults to ``data/raw/``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(collisions, casualties, vehicles)`` DataFrames.

    Raises
    ------
    FileNotFoundError
        If any of the three expected CSV files is missing.
    """
    if raw_dir is None:
        raw_dir = data_dir("raw")

    tables = {}
    for key, fname in _FILENAMES.items():
        path = raw_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Missing STATS19 file: {path}\n"
                "Download from https://www.gov.uk/government/statistical-data-sets/road-safety-open-data"
            )
        tables[key] = pd.read_csv(path, low_memory=False)
        logger.info("%s loaded: %s rows × %s cols", key, *tables[key].shape)

    return tables["collisions"], tables["casualties"], tables["vehicles"]


def _collapse_casualties(casualties: pd.DataFrame) -> pd.DataFrame:
    """Keep the most severely injured casualty per collision.

    Sorts by ``casualty_severity`` ascending (1=Fatal is the most severe),
    then groups by ``accident_index`` and keeps the first row.

    Parameters
    ----------
    casualties : pd.DataFrame
        Raw casualties table.

    Returns
    -------
    pd.DataFrame
        One row per ``accident_index``, representing the worst outcome.
    """
    return (
        casualties
        .sort_values("casualty_severity")
        .groupby("accident_index", as_index=False)
        .first()
    )


def _collapse_vehicles(vehicles: pd.DataFrame) -> pd.DataFrame:
    """Keep the first reported vehicle per collision.

    This is a deliberate simplification for multi-vehicle crashes.
    The limitation is documented in Notebook 1.

    Parameters
    ----------
    vehicles : pd.DataFrame
        Raw vehicles table.

    Returns
    -------
    pd.DataFrame
        One row per ``accident_index``.
    """
    return vehicles.groupby("accident_index", as_index=False).first()


def build_london_dataset(
    collisions: pd.DataFrame,
    casualties: pd.DataFrame,
    vehicles: pd.DataFrame,
) -> pd.DataFrame:
    """Merge and filter the three tables to produce a London collision dataset.

    Steps
    -----
    1. Collapse casualties to worst per collision.
    2. Collapse vehicles to first per collision.
    3. Left-merge casualties onto collisions.
    4. Left-merge vehicles onto collisions.
    5. Filter to Greater London (police_force ∈ {1, 48}).

    Parameters
    ----------
    collisions : pd.DataFrame
    casualties : pd.DataFrame
    vehicles   : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        London-only collision dataset, one row per collision.
    """
    casualties_worst = _collapse_casualties(casualties)
    vehicles_first = _collapse_vehicles(vehicles)

    df = collisions.merge(casualties_worst, on="accident_index", how="left", suffixes=("", "_cas"))
    df = df.merge(vehicles_first, on="accident_index", how="left", suffixes=("", "_veh"))

    london_df = df[df["police_force"].isin(_LONDON_FORCES)].copy()
    logger.info(
        "London dataset: %s rows × %s cols (%.1f%% of UK total)",
        len(london_df),
        london_df.shape[1],
        100 * len(london_df) / len(collisions),
    )
    return london_df


def load_london_dataset(raw_dir: Path | None = None) -> pd.DataFrame:
    """Convenience wrapper: load raw tables, join, and return London dataset.

    Parameters
    ----------
    raw_dir : Path, optional
        Directory containing the CSV files.

    Returns
    -------
    pd.DataFrame
        Ready-to-use London collision DataFrame.
    """
    collisions, casualties, vehicles = load_raw_tables(raw_dir)
    return build_london_dataset(collisions, casualties, vehicles)
