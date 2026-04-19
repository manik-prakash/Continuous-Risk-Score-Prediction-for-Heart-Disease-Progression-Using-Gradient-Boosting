"""Model explainability: SHAP, feature importance, and PDP plots."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from heart_risk.config import FIGURES_DIR, ALL_FEATURES
from heart_risk.preprocess import get_feature_names

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str) -> Path:
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def plot_xgb_feature_importance(pipeline, feature_names: list[str]) -> Path:
    try:
        regressor = pipeline.named_steps["regressor"]
        importances = regressor.feature_importances_
    except AttributeError:
        print("  Model does not expose feature_importances_ — skipping.")
        return None
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(fi) // 3)))
    fi.plot.barh(ax=ax, color="steelblue")
    ax.set_title("Feature Importance (Gain)")
    ax.set_xlabel("Importance")
    return _save(fig, "07_feature_importance.png")


def plot_shap_summary(pipeline, X_transformed: np.ndarray, feature_names: list[str]) -> Path:
    try:
        import shap
    except ImportError:
        print("  shap not installed — skipping SHAP plots.")
        return None
    regressor = pipeline.named_steps["regressor"]
    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X_transformed)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    return _save(plt.gcf(), "08_shap_summary.png")


def plot_shap_waterfall(pipeline, X_transformed: np.ndarray, feature_names: list[str],
                        patient_idx: int = 0) -> Path:
    try:
        import shap
    except ImportError:
        return None
    regressor = pipeline.named_steps["regressor"]
    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer(X_transformed)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap_values[patient_idx], show=False)
    return _save(plt.gcf(), "09_shap_waterfall.png")


def plot_shap_dependence(pipeline, X_transformed: np.ndarray, feature_names: list[str],
                         top_n: int = 4) -> list[Path]:
    try:
        import shap
    except ImportError:
        return []
    regressor = pipeline.named_steps["regressor"]
    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X_transformed)

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[-top_n:][::-1]
    paths = []
    for idx in top_indices:
        fname = feature_names[idx].replace("/", "_").replace(" ", "_")
        fig, ax = plt.subplots(figsize=(7, 5))
        shap.dependence_plot(idx, shap_values, X_transformed,
                             feature_names=feature_names, ax=ax, show=False)
        paths.append(_save(fig, f"10_shap_dep_{fname}.png"))
    return paths


def plot_pdp(pipeline, X_train, feature_names: list[str]) -> Path:
    from sklearn.inspection import PartialDependenceDisplay
    pdp_features = ["oldpeak", "thalach", "ca", "cp"]
    indices = [feature_names.index(f) for f in pdp_features if f in feature_names]
    if not indices:
        return None
    regressor = pipeline.named_steps["regressor"]
    preprocessor = pipeline.named_steps["preprocessor"]
    X_t = preprocessor.transform(X_train)

    fig, axes = plt.subplots(1, len(indices), figsize=(5 * len(indices), 4))
    if len(indices) == 1:
        axes = [axes]
    PartialDependenceDisplay.from_estimator(
        regressor, X_t, indices, feature_names=feature_names,
        ax=axes, grid_resolution=50,
    )
    fig.suptitle("Partial Dependence Plots", fontweight="bold")
    return _save(fig, "11_pdp.png")


def run_explainability(pipeline, X_train, X_test) -> None:
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names(preprocessor)
    X_test_t = preprocessor.transform(X_test)
    X_train_t = preprocessor.transform(X_train)

    plot_xgb_feature_importance(pipeline, feature_names)
    plot_shap_summary(pipeline, X_test_t, feature_names)
    plot_shap_waterfall(pipeline, X_test_t, feature_names, patient_idx=0)
    plot_shap_dependence(pipeline, X_test_t, feature_names)
    plot_pdp(pipeline, X_train, feature_names)
    print("Explainability plots saved.")
