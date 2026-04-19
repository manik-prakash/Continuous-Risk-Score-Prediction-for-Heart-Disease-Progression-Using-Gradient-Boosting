"""Reusable preprocessing pipeline built with ColumnTransformer."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from heart_risk.config import (
    CONTINUOUS_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES,
    DISCRETE_NUMERIC, ALL_FEATURES, TARGET, TARGET_RAW,
    TEST_SIZE, RANDOM_SEED,
)
from sklearn.model_selection import train_test_split


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for the heart-risk feature set."""

    continuous_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    binary_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    discrete_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("continuous", continuous_pipe, CONTINUOUS_FEATURES),
            ("binary", binary_pipe, BINARY_FEATURES),
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
            ("discrete", discrete_pipe, DISCRETE_NUMERIC),
        ],
        remainder="drop",
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract feature names after fitting the ColumnTransformer."""
    names: list[str] = []
    for name, pipe, cols in preprocessor.transformers_:
        if name == "categorical":
            ohe = pipe.named_steps["ohe"]
            names.extend(ohe.get_feature_names_out(cols).tolist())
        else:
            names.extend(cols if isinstance(cols, list) else list(cols))
    return names


def split_cleveland(df: pd.DataFrame):
    """Stratified train/test split on original `num` label."""
    X = df[ALL_FEATURES]
    y = df[TARGET]
    strat = df[TARGET_RAW]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=strat
    )
    return X_train, X_test, y_train, y_test
