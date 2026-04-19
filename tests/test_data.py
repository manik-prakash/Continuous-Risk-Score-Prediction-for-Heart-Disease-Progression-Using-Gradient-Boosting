"""Unit tests for data loading."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import numpy as np
from heart_risk.data import load_cleveland
from heart_risk.config import TARGET, TARGET_RAW


def test_cleveland_shape():
    df = load_cleveland()
    assert df.shape == (303, 15), f"Expected (303, 15), got {df.shape}"  # 14 cols + risk_score


def test_no_question_marks():
    df = load_cleveland()
    for col in df.columns:
        assert "?" not in df[col].astype(str).values, f"'?' found in column {col}"


def test_risk_score_range():
    df = load_cleveland()
    assert df[TARGET].between(0.0, 1.0).all(), "risk_score outside [0, 1]"


def test_risk_score_formula():
    df = load_cleveland()
    expected = df[TARGET_RAW] / 4.0
    assert (df[TARGET] == expected).all(), "risk_score != num / 4"
