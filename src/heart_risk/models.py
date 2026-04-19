"""Model definitions, cross-validation, hyperparameter tuning, and artifact saving."""

import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error

from heart_risk.config import (
    RANDOM_SEED, CV_FOLDS, RANDOM_SEARCH_ITER,
    XGB_PARAM_GRID, LGBM_PARAM_GRID, GBR_PARAM_GRID,
    BEST_MODEL_PATH, MODELS_DIR,
)
from heart_risk.preprocess import build_preprocessor

MODELS_DIR.mkdir(parents=True, exist_ok=True)

mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)


def _make_pipeline(regressor) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", regressor),
    ])


def _cv_summary(cv_results: dict, name: str) -> dict:
    return {
        "model": name,
        "cv_mae": -cv_results["test_neg_mae"].mean(),
        "cv_mae_std": cv_results["test_neg_mae"].std(),
        "cv_rmse": np.sqrt((-cv_results["test_neg_mse"]).mean()) if "test_neg_mse" in cv_results else None,
        "cv_r2": cv_results["test_r2"].mean(),
    }


def train_baselines(X_train, y_train) -> list[dict]:
    """Train LinearRegression, Ridge, RandomForest with 5-fold CV."""
    baselines = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=RANDOM_SEED),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED),
    }
    results = []
    fitted = {}
    for name, reg in baselines.items():
        pipe = _make_pipeline(reg)
        cv = cross_validate(
            pipe, X_train, y_train, cv=CV_FOLDS,
            scoring={"neg_mae": mae_scorer, "neg_mse": "neg_mean_squared_error", "r2": "r2"},
            return_train_score=False,
        )
        results.append(_cv_summary(cv, name))
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        print(f"  {name}: MAE={results[-1]['cv_mae']:.4f} ± {results[-1]['cv_mae_std']:.4f}")
    return results, fitted


def _tune_model(name: str, regressor, param_grid: dict, X_train, y_train) -> tuple:
    pipe = _make_pipeline(regressor)
    search = RandomizedSearchCV(
        pipe, param_grid,
        n_iter=RANDOM_SEARCH_ITER, cv=CV_FOLDS,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_SEED, n_jobs=-1, refit=True,
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    best_mae = -search.best_score_
    print(f"  {name}: best CV MAE={best_mae:.4f}  params={search.best_params_}")
    return search.best_estimator_, best_mae, search.best_params_


def train_gradient_boosting(X_train, y_train) -> list[dict]:
    """Tune XGBoost, LightGBM, and GradientBoostingRegressor."""
    results = []
    fitted = {}

    # XGBoost
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(random_state=RANDOM_SEED, verbosity=0, n_jobs=-1)
        pipe, mae, params = _tune_model("XGBoost", xgb, XGB_PARAM_GRID, X_train, y_train)
        results.append({"model": "XGBoost", "cv_mae": mae, "best_params": params})
        fitted["XGBoost"] = pipe
    except ImportError:
        print("  XGBoost not installed — skipping.")

    # LightGBM
    try:
        from lightgbm import LGBMRegressor
        lgbm = LGBMRegressor(random_state=RANDOM_SEED, verbose=-1, n_jobs=-1)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            pipe, mae, params = _tune_model("LightGBM", lgbm, LGBM_PARAM_GRID, X_train, y_train)
        results.append({"model": "LightGBM", "cv_mae": mae, "best_params": params})
        fitted["LightGBM"] = pipe
    except ImportError:
        print("  LightGBM not installed — skipping.")

    # GradientBoostingRegressor (sklearn, always available)
    gbr = GradientBoostingRegressor(random_state=RANDOM_SEED)
    pipe, mae, params = _tune_model("GradientBoosting", gbr, GBR_PARAM_GRID, X_train, y_train)
    results.append({"model": "GradientBoosting", "cv_mae": mae, "best_params": params})
    fitted["GradientBoosting"] = pipe

    return results, fitted


def select_best(baseline_results, gb_results, baseline_fitted, gb_fitted) -> tuple:
    """Return (name, pipeline) of the model with lowest CV MAE."""
    all_results = baseline_results + gb_results
    all_fitted = {**baseline_fitted, **gb_fitted}
    best = min(all_results, key=lambda r: r["cv_mae"])
    best_name = best["model"]
    print(f"\nBest model: {best_name}  (CV MAE={best['cv_mae']:.4f})")
    return best_name, all_fitted[best_name], all_results


def save_pipeline(pipeline: Pipeline, path=BEST_MODEL_PATH) -> None:
    joblib.dump(pipeline, path)
    print(f"Model saved to {path}")


def load_pipeline(path=BEST_MODEL_PATH) -> Pipeline:
    return joblib.load(path)
