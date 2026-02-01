# Advanced Ensemble U-MIDAS (Python package)

This repository implements a walk-forward (expanding window) **ensemble U-MIDAS** pipeline for panel forecasting of credit spread levels.

## What the package provides

- **Mixed-frequency U-MIDAS feature construction**
  - Autoregressive (AR) lags of the dependent variable level (`cs`)
  - Monthly predictors as a fixed-length lag window
  - Quarterly predictors as publication-lagged “taps” spaced 3 months apart
- **Walk-forward training and prediction**
  - Train on all data up to month-end `t`
  - Predict `cs` at month-end `t+1` (or more generally `t+H`)
- **Ensemble modelling**
  - Base learners: ElasticNet, Ridge, (LightGBM or XGBoost) gradient boosting, Random Forest
  - Meta-learner: stacking ridge (default), with optional alternatives
- **Evaluation + diagnostics**
  - Micro and macro (per-bond averaged) metrics
  - Diagnostic plots (scatter, residuals, temporal RMSE, tail-risk diagnostics, etc.)

> Note: The core methodology and logic are preserved from the notebook.  
> This refactor primarily improves structure, parameterisation, reproducibility, and documentation.

## Installation

From the repository root:

```bash
pip install -e .
```

Optional extras:

```bash
# For LightGBM / XGBoost
pip install -e ".[boosting]"

# For documentation builds
pip install -e ".[docs]"

# For development tooling
pip install -e ".[dev]"
```

## Data expectations

The panel CSV must contain at least:

- `cusip` (bond identifier)
- `date` (timestamp)
- `cs` (credit spread level, in decimal units)

An optional Excel variable dictionary can be provided to infer which columns are monthly vs quarterly predictors.

## Quickstart

### 1) Run the walk-forward ensemble

```bash
umidas-ensemble run   --data path/to/final_data_modified.csv   --dict path/to/variable_dictionary.xlsx   --save-dir results   --horizons 1   --meta stacking_ridge   --feature-engineering enhanced
```

This creates a prediction file per horizon:

- `results/u_midas_advanced_ensemble_lead{H}_predictions.csv`

### 2) Evaluate predictions and generate plots

```bash
umidas-ensemble evaluate   --predictions results/u_midas_advanced_ensemble_lead1_predictions.csv   --horizon 1   --data path/to/final_data_modified.csv   --save-dir results   --plot ensemble_performance_analysis.png
```

## Baseline evaluation

The original notebook’s naive baseline merged on the *same* month as `y_true`, which makes the baseline error artificially zero.  
This package implements the correct **random-walk / no-change baseline** aligned with horizon `H`:

- baseline prediction for month `t` is `cs_{t-H}`.

## Documentation

A Read the Docs-compatible Sphinx site is included in `docs/`.

Build locally:

```bash
pip install -e ".[docs]"
cd docs
make html
```

The rendered documentation includes:

- full methodology explanation
- API reference
- notebook-derived results summary and plots
