"""Unit tests for evaluation utilities."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from heart_risk.evaluate import compute_metrics, to_risk_bands, binary_auc
from heart_risk.config import RISK_BAND_LABELS


def test_metrics_return_numeric():
    y_true = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    y_pred = np.array([0.05, 0.20, 0.55, 0.70, 0.95])
    metrics = compute_metrics(y_true, y_pred)
    for k, v in metrics.items():
        assert isinstance(v, float), f"Metric {k} is not float: {type(v)}"


def test_risk_band_boundaries():
    scores = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    bands = list(to_risk_bands(scores))
    assert bands[0] == "Low"
    assert bands[1] == "Moderate"
    assert bands[2] == "High"
    assert bands[3] == "Very High"
    assert bands[4] == "Very High"  # 1.0 is in last bin


def test_risk_band_all_labels_covered():
    scores = np.array([0.1, 0.3, 0.6, 0.9])
    bands = list(to_risk_bands(scores))
    assert bands == ["Low", "Moderate", "High", "Very High"]


def test_binary_auc_normal():
    y_raw = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0.0, 0.1, 0.6, 0.7, 0.9])
    result = binary_auc(y_raw, y_pred)
    assert isinstance(result, float), f"Expected float, got {type(result)}"
    assert 0.0 <= result <= 1.0


def test_binary_auc_single_class():
    y_raw = np.array([0, 0, 0])
    y_pred = np.array([0.1, 0.2, 0.3])
    result = binary_auc(y_raw, y_pred)
    assert isinstance(result, str) and "N/A" in result
