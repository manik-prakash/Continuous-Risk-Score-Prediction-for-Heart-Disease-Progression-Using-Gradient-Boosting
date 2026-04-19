"""Risk profile visualizations: scatter, residuals, demographics, progression."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from heart_risk.config import FIGURES_DIR, ALL_FEATURES, TARGET, TARGET_RAW
from heart_risk.evaluate import clip_predictions, to_risk_bands

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str) -> Path:
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def plot_pred_vs_actual(y_true, y_pred, tag: str = "test") -> Path:
    y_pred = clip_predictions(np.asarray(y_pred))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.4, s=50)
    lims = [0, 1]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual risk_score")
    ax.set_ylabel("Predicted risk_score")
    ax.set_title(f"Predicted vs Actual — {tag}")
    ax.legend()
    return _save(fig, f"12_pred_vs_actual_{tag}.png")


def plot_residuals(y_true, y_pred, tag: str = "test") -> Path:
    y_pred = clip_predictions(np.asarray(y_pred))
    residuals = np.asarray(y_true) - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.4, s=50)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted risk_score")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Fitted")
    axes[1].hist(residuals, bins=20, color="steelblue", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual Distribution")
    fig.suptitle(f"Residual Analysis — {tag}", fontweight="bold")
    return _save(fig, f"13_residuals_{tag}.png")


def plot_risk_by_demographics(df: pd.DataFrame, y_pred: np.ndarray) -> Path:
    """Risk distribution by age band and sex."""
    df = df.copy()
    df["pred_risk"] = clip_predictions(y_pred)
    df["age_band"] = pd.cut(df["age"], bins=[0, 40, 50, 60, 70, 120],
                            labels=["<40", "40–50", "50–60", "60–70", "70+"])
    df["sex_label"] = df["sex"].map({0: "Female", 1: "Male"})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # By age band
    order = ["<40", "40–50", "50–60", "60–70", "70+"]
    age_data = df.groupby("age_band", observed=True)["pred_risk"].mean().reindex(order)
    axes[0].bar(age_data.index, age_data.values, color=sns.color_palette("Oranges", len(order)))
    axes[0].set_title("Mean Predicted Risk by Age Band")
    axes[0].set_xlabel("Age Band")
    axes[0].set_ylabel("Mean Predicted Risk Score")
    axes[0].set_ylim(0, 1)

    # By sex
    sex_data = df.groupby("sex_label")["pred_risk"].mean()
    axes[1].bar(sex_data.index, sex_data.values, color=["#5DB0D7", "#F08030"])
    axes[1].set_title("Mean Predicted Risk by Sex")
    axes[1].set_xlabel("Sex")
    axes[1].set_ylabel("Mean Predicted Risk Score")
    axes[1].set_ylim(0, 1)

    fig.suptitle("Predicted Risk by Demographics", fontweight="bold")
    return _save(fig, "14_risk_by_demographics.png")


def progression_table(pipeline, df_cleveland_train: pd.DataFrame) -> pd.DataFrame:
    """Show risk change as ca increases 0→3 using median patient profile."""
    median_row = df_cleveland_train[ALL_FEATURES].median(numeric_only=True)
    rows = []
    for ca_val in [0, 1, 2, 3]:
        row = median_row.copy()
        row["ca"] = ca_val
        X = pd.DataFrame([row])
        risk = clip_predictions(pipeline.predict(X))[0]
        rows.append({"ca": ca_val, "predicted_risk_score": round(risk, 4),
                     "risk_band": to_risk_bands(np.array([risk]))[0]})
    df_prog = pd.DataFrame(rows)
    print("\nProgression Table (median patient, varying ca):")
    print(df_prog.to_string(index=False))
    return df_prog
