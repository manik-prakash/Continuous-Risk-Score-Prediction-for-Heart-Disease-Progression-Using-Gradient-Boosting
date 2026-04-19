"""Data loading utilities for UCI Heart Disease processed files."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union

from heart_risk.config import COLUMNS, TARGET_RAW, TARGET, DATA_FILES


def load_uci_file(path: Union[str, Path]) -> pd.DataFrame:
    """Load a single UCI processed heart-disease file.

    Handles '?' missing-value placeholders and coerces all columns to numeric
    except where values are genuinely categorical.
    """
    path = Path(path)
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")

    # Ensure numeric dtype throughout; non-parseable strings become NaN
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET] = df[TARGET_RAW] / 4.0
    return df


def load_cleveland() -> pd.DataFrame:
    return load_uci_file(DATA_FILES["cleveland"])


def load_all() -> dict[str, pd.DataFrame]:
    """Load all four hospital datasets, keyed by hospital name."""
    return {name: load_uci_file(path) for name, path in DATA_FILES.items()}


def summary(df: pd.DataFrame, name: str = "dataset") -> None:
    print(f"\n{'='*50}")
    print(f"  {name.upper()}")
    print(f"{'='*50}")
    print(f"  Shape : {df.shape}")
    print(f"  Target distribution (num):\n{df[TARGET_RAW].value_counts().sort_index().to_string()}")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(f"  Missing values:\n{missing.to_string()}")
    else:
        print("  No missing values.")
