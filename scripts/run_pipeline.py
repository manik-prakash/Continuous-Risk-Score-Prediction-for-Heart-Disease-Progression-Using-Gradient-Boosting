"""Command-line runner: executes the full heart-risk pipeline end-to-end."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heart_risk.data import load_cleveland, load_all, summary
from heart_risk.eda import run_full_eda
from heart_risk.preprocess import split_cleveland
from heart_risk.models import train_baselines, train_gradient_boosting, select_best, save_pipeline
from heart_risk.evaluate import evaluate_split, evaluate_external, save_metrics, clip_predictions
from heart_risk.explain import run_explainability
from heart_risk.visualize import plot_pred_vs_actual, plot_residuals, plot_risk_by_demographics, progression_table
from heart_risk.config import ALL_FEATURES, FIGURES_DIR, METRICS_DIR, REPORTS_DIR


def main():
    print("=" * 60)
    print("  Heart Disease Continuous Risk Score Pipeline")
    print("=" * 60)

    # Load
    df_cleveland = load_cleveland()
    all_dfs = load_all()
    external_dfs = {k: v for k, v in all_dfs.items() if k != "cleveland"}
    summary(df_cleveland, "Cleveland")

    # EDA
    print("\n[1/6] Running EDA...")
    run_full_eda(df_cleveland)

    # Split
    print("\n[2/6] Splitting Cleveland train/test...")
    X_train, X_test, y_train, y_test = split_cleveland(df_cleveland)

    # Train
    print("\n[3/6] Training baseline models...")
    baseline_results, baseline_fitted = train_baselines(X_train, y_train)

    print("\n[4/6] Tuning gradient boosting models...")
    gb_results, gb_fitted = train_gradient_boosting(X_train, y_train)

    best_name, best_pipeline, all_results = select_best(
        baseline_results, gb_results, baseline_fitted, gb_fitted
    )
    save_pipeline(best_pipeline)

    # Evaluate
    print("\n[5/6] Evaluating...")
    df_test_raw = df_cleveland.loc[X_test.index]
    test_metrics = evaluate_split(best_pipeline, X_test, y_test, df_test_raw, tag="cleveland_test")
    save_metrics(test_metrics, "cleveland_test_metrics.json")

    external_metrics = evaluate_external(best_pipeline, external_dfs)
    for hospital, metrics in external_metrics.items():
        save_metrics(metrics, f"{hospital}_metrics.json")

    # Visualize
    print("\n[6/6] Generating visualizations...")
    y_pred_test = clip_predictions(best_pipeline.predict(X_test))
    plot_pred_vs_actual(y_test, y_pred_test, tag="cleveland_test")
    plot_residuals(y_test, y_pred_test, tag="cleveland_test")

    y_pred_all = clip_predictions(best_pipeline.predict(df_cleveland[ALL_FEATURES]))
    plot_risk_by_demographics(df_cleveland, y_pred_all)
    progression_table(best_pipeline, df_cleveland.loc[X_train.index])

    # Explainability
    run_explainability(best_pipeline, X_train, X_test)

    # Summary report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "best_model": best_name,
        "cleveland_test_metrics": test_metrics,
        "external_metrics": external_metrics,
        "all_cv_results": all_results,
    }
    report_path = REPORTS_DIR / "pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nPipeline complete. Report: {report_path}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
