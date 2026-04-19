# Heart Disease Risk Score Predictor

A machine learning project that predicts **how likely someone is to have heart disease**, as a continuous score between 0 and 1.

Built using the [UCI Heart Disease dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease) (Cleveland clinic data).

---

## What does it do?

The UCI dataset has a column called `num` that goes from 0 (no disease) to 4 (severe disease).  
We convert it into a **risk score** using a simple formula:

```
risk_score = num / 4
```

So `0` means no risk, `1` means maximum risk. We then train regression models to predict this score from patient features like age, chest pain type, cholesterol, etc.

---

## Project Structure

```
AI-IA-3/
├── data/                        # Raw UCI data files (already downloaded)
├── notebooks/
│   └── 01_heart_risk_regression.ipynb   # Main notebook (start here)
├── src/heart_risk/
│   ├── config.py                # All settings in one place
│   ├── data.py                  # Load and clean the data
│   ├── eda.py                   # Exploratory plots
│   ├── preprocess.py            # Feature engineering pipeline
│   ├── models.py                # Train and tune ML models
│   ├── evaluate.py              # Metrics and confusion matrix
│   ├── explain.py               # SHAP feature importance plots
│   └── visualize.py             # Risk charts and demographic plots
├── scripts/
│   └── run_pipeline.py          # Run everything from command line
├── tests/                       # Unit tests
├── outputs/                     # Figures, metrics, saved model (auto-generated)
└── requirements.txt
```

---

## How to Run

### 1. Set up the virtual environment

```bash
python -m venv heart_risk_env

# Windows
heart_risk_env\Scripts\activate

# Mac / Linux
source heart_risk_env/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline

```bash
python scripts/run_pipeline.py
```

This trains all models, saves plots to `outputs/figures/`, and writes metrics to `outputs/metrics/`.

### 4. Or open the notebook

```bash
jupyter notebook notebooks/01_heart_risk_regression.ipynb
```

---

## Models Used

| Model | Type |
|---|---|
| Linear Regression | Baseline |
| Ridge Regression | Baseline |
| Random Forest | Baseline |
| XGBoost | Gradient Boosting |
| LightGBM | Gradient Boosting |
| Gradient Boosting (sklearn) | Gradient Boosting |

The best model is selected automatically based on cross-validated MAE.

---

## Risk Bands

| Score Range | Band |
|---|---|
| 0.00 – 0.25 | Low |
| 0.25 – 0.50 | Moderate |
| 0.50 – 0.75 | High |
| 0.75 – 1.00 | Very High |

---

## Run Tests

```bash
python -m pytest tests/ -v
```

13 unit tests covering data loading, preprocessing, and evaluation logic.

---

## Dataset

- **Cleveland** (303 patients) — used for training and testing
- **Hungarian, Switzerland, VA Long Beach** — used for external validation only

> Note: The Hungarian processed file only contains `num = 0` or `num = 1`, so its risk score range is narrower than Cleveland's. External metrics should be read with that in mind.

---

## Key Results (Cleveland Test Set)

| Metric | Value |
|---|---|
| MAE | ~0.186 |
| RMSE | ~0.241 |
| R² | ~0.40 |
| AUC-ROC (binary) | ~0.83 |

---

## Requirements

- Python 3.10+
- See `requirements.txt` for full list
