"""Exploratory data analysis plots and summaries."""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

from heart_risk.config import (
    CONTINUOUS_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES,
    DISCRETE_NUMERIC, TARGET_RAW, TARGET, FIGURES_DIR,
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def plot_target_distribution(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = df[TARGET_RAW].value_counts().sort_index()
    axes[0].bar(counts.index.astype(str), counts.values, color=sns.color_palette("Blues_d", len(counts)))
    axes[0].set_title("Raw num target distribution")
    axes[0].set_xlabel("num (0–4)")
    axes[0].set_ylabel("Count")
    for bar, v in zip(axes[0].patches, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(v), ha="center", fontsize=10)

    axes[1].hist(df[TARGET], bins=20, color="steelblue", edgecolor="white")
    axes[1].set_title("risk_score = num / 4 distribution")
    axes[1].set_xlabel("risk_score")
    axes[1].set_ylabel("Count")
    fig.suptitle("Heart Disease Target Distribution", fontweight="bold")
    return _save(fig, "01_target_distribution.png")


def plot_continuous_histograms(df: pd.DataFrame) -> Path:
    cols = CONTINUOUS_FEATURES
    fig, axes = plt.subplots(1, len(cols), figsize=(18, 4))
    for ax, col in zip(axes, cols):
        ax.hist(df[col].dropna(), bins=20, color="teal", edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel(col)
    fig.suptitle("Continuous Feature Distributions", fontweight="bold")
    return _save(fig, "02_continuous_histograms.png")


def plot_continuous_boxplots(df: pd.DataFrame) -> Path:
    cols = CONTINUOUS_FEATURES
    fig, axes = plt.subplots(1, len(cols), figsize=(18, 4))
    for ax, col in zip(axes, cols):
        ax.boxplot(df[col].dropna(), patch_artist=True,
                   boxprops=dict(facecolor="lightblue", color="navy"),
                   medianprops=dict(color="red", linewidth=2))
        ax.set_title(col)
        ax.set_xlabel("")
    fig.suptitle("Continuous Feature Boxplots", fontweight="bold")
    return _save(fig, "03_continuous_boxplots.png")


def plot_categorical_counts(df: pd.DataFrame) -> Path:
    cols = CATEGORICAL_FEATURES + BINARY_FEATURES + DISCRETE_NUMERIC
    n = len(cols)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(16, 8))
    axes = axes.flatten()
    palette = sns.color_palette("Set2")
    for ax, col in zip(axes, cols):
        counts = df[col].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color=palette[: len(counts)])
        ax.set_title(col)
        ax.set_xlabel(col)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Categorical / Discrete Feature Distributions", fontweight="bold")
    return _save(fig, "04_categorical_counts.png")


def plot_missing_values(df: pd.DataFrame) -> Path:
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    fig, ax = plt.subplots(figsize=(8, 4))
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=14)
    else:
        ax.barh(missing.index, missing.values, color="salmon")
        ax.set_xlabel("Missing count")
        ax.set_title("Missing Values per Column")
        for i, v in enumerate(missing.values):
            ax.text(v + 0.3, i, str(v), va="center")
    return _save(fig, "05_missing_values.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap", fontweight="bold")
    return _save(fig, "06_correlation_heatmap.png")


def run_full_eda(df: pd.DataFrame) -> list[Path]:
    """Run all EDA plots, return list of saved paths."""
    paths = [
        plot_target_distribution(df),
        plot_continuous_histograms(df),
        plot_continuous_boxplots(df),
        plot_categorical_counts(df),
        plot_missing_values(df),
        plot_correlation_heatmap(df),
    ]
    print(f"EDA plots saved to {FIGURES_DIR}")
    return paths
