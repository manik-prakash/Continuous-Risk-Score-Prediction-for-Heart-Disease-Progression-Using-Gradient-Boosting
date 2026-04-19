"""Evaluation metrics, risk-band classification, and external hospital validation."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, roc_auc_score,
)
from scipy.stats import spearmanr

from heart_risk.config import (
    ALL_FEATURES, TARGET, TARGET_RAW,
    RISK_BAND_BINS, RISK_BAND_LABELS,
    FIGURES_DIR, METRICS_DIR,
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def clip_predictions(y_pred: np.ndarray) -> np.ndarray:
    return np.clip(y_pred, 0.0, 1.0)


def compute_metrics(y_true, y_pred) -> dict:
    y_pred = clip_predictions(np.asarray(y_pred))
    y_true = np.asarray(y_true)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    rho, pval = spearmanr(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Spearman_rho": rho, "Spearman_p": pval}


def to_risk_bands(scores: np.ndarray) -> pd.Categorical:
    scores = clip_predictions(np.asarray(scores))
    return pd.cut(scores, bins=RISK_BAND_BINS, labels=RISK_BAND_LABELS, right=False)


def risk_band_confusion_matrix(y_true, y_pred, tag: str = "cleveland_test") -> pd.DataFrame:
    true_bands = to_risk_bands(np.asarray(y_true))
    pred_bands = to_risk_bands(clip_predictions(np.asarray(y_pred)))
    cm = confusion_matrix(true_bands, pred_bands, labels=RISK_BAND_LABELS)
    df_cm = pd.DataFrame(cm, index=RISK_BAND_LABELS, columns=RISK_BAND_LABELS)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=RISK_BAND_LABELS, yticklabels=RISK_BAND_LABELS)
    ax.set_xlabel("Predicted Band")
    ax.set_ylabel("True Band")
    ax.set_title(f"Risk-Band Confusion Matrix — {tag}")
    path = FIGURES_DIR / f"cm_{tag}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return df_cm


def binary_auc(y_true_raw: np.ndarray, y_pred: np.ndarray) -> float | str:
    """AUC-ROC: num==0 (no disease) vs num>0 (disease)."""
    binary = (np.asarray(y_true_raw) > 0).astype(int)
    if len(np.unique(binary)) < 2:
        return "N/A (single class)"
    try:
        return roc_auc_score(binary, clip_predictions(np.asarray(y_pred)))
    except Exception as e:
        return f"N/A ({e})"


def evaluate_split(pipeline, X_test, y_test, df_test_raw, tag: str = "cleveland_test") -> dict:
    y_pred = clip_predictions(pipeline.predict(X_test))
    metrics = compute_metrics(y_test, y_pred)
    metrics["AUC_ROC"] = binary_auc(df_test_raw[TARGET_RAW], y_pred)
    print(f"\n[{tag}] Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    risk_band_confusion_matrix(y_test, y_pred, tag=tag)
    return metrics


def evaluate_external(pipeline, hospital_dfs: dict) -> dict[str, dict]:
    """Run evaluation on each external hospital dataset."""
    all_metrics = {}
    for hospital, df in hospital_dfs.items():
        df = df.dropna(subset=[TARGET_RAW])
        if df.empty:
            print(f"  [{hospital}] No valid rows — skipping.")
            continue
        X = df[ALL_FEATURES]
        y = df[TARGET]
        try:
            y_pred = clip_predictions(pipeline.predict(X))
            metrics = compute_metrics(y, y_pred)
            metrics["AUC_ROC"] = binary_auc(df[TARGET_RAW], y_pred)
            all_metrics[hospital] = metrics
            print(f"\n[{hospital}] Metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
            risk_band_confusion_matrix(y, y_pred, tag=hospital)
        except Exception as e:
            print(f"  [{hospital}] Error: {e}")
    return all_metrics


def save_metrics(metrics: dict, filename: str = "metrics.json") -> None:
    import json
    path = METRICS_DIR / filename
    serialisable = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in metrics.items()}
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Metrics saved to {path}")
