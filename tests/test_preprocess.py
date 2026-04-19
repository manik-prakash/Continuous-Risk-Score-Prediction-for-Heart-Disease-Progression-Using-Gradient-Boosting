"""Unit tests for preprocessing pipeline."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from heart_risk.data import load_cleveland
from heart_risk.preprocess import build_preprocessor, split_cleveland, get_feature_names


def test_no_missing_after_transform():
    df = load_cleveland()
    X_train, X_test, _, _ = split_cleveland(df)
    prep = build_preprocessor()
    prep.fit(X_train)
    X_train_t = prep.transform(X_train)
    X_test_t = prep.transform(X_test)
    assert not np.isnan(X_train_t).any(), "NaN in train after transform"
    assert not np.isnan(X_test_t).any(), "NaN in test after transform"


def test_matching_feature_dimensions():
    df = load_cleveland()
    X_train, X_test, _, _ = split_cleveland(df)
    prep = build_preprocessor()
    prep.fit(X_train)
    assert prep.transform(X_train).shape[1] == prep.transform(X_test).shape[1]


def test_feature_names_stable():
    df = load_cleveland()
    X_train, _, _, _ = split_cleveland(df)
    prep = build_preprocessor()
    prep.fit(X_train)
    names1 = get_feature_names(prep)
    names2 = get_feature_names(prep)
    assert names1 == names2, "Feature names not stable across calls"


def test_ohe_feature_names_present():
    df = load_cleveland()
    X_train, _, _, _ = split_cleveland(df)
    prep = build_preprocessor()
    prep.fit(X_train)
    names = get_feature_names(prep)
    # Should contain OHE columns for cp
    assert any("cp" in n for n in names), "No OHE features for cp"
