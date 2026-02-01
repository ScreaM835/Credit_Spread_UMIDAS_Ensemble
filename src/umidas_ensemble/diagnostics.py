from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EnsembleDiagnostics:
    n_predictions: int
    n_bonds: int
    date_min: pd.Timestamp
    date_max: pd.Timestamp

    pred_desc: Dict[str, float]
    true_desc: Dict[str, float]

    range_ratio: float
    correlation: float

    mean_bias: float
    rmse: float
    residual_std: float

    yearly_rmse: Dict[int, float]
    bond_rmse_desc: Dict[str, float]
    outlier_bonds: Dict[str, float]

    prediction_uniqueness_ratio: float

    error_desc: Dict[str, float]
    fat_tail_ratio_95_50: float

    issues: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_predictions": self.n_predictions,
            "n_bonds": self.n_bonds,
            "date_min": str(self.date_min.date()),
            "date_max": str(self.date_max.date()),
            "pred_desc": self.pred_desc,
            "true_desc": self.true_desc,
            "range_ratio": self.range_ratio,
            "correlation": self.correlation,
            "mean_bias": self.mean_bias,
            "rmse": self.rmse,
            "residual_std": self.residual_std,
            "yearly_rmse": self.yearly_rmse,
            "bond_rmse_desc": self.bond_rmse_desc,
            "outlier_bonds": self.outlier_bonds,
            "prediction_uniqueness_ratio": self.prediction_uniqueness_ratio,
            "error_desc": self.error_desc,
            "fat_tail_ratio_95_50": self.fat_tail_ratio_95_50,
            "issues": list(self.issues),
        }


def diagnose_predictions(pred: pd.DataFrame) -> EnsembleDiagnostics:
    """Compute a set of diagnostic summaries for model predictions."""
    required = {"cusip", "date", "y_true", "y_pred"}
    missing = sorted(required - set(pred.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = pred.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["cusip"] = df["cusip"].astype(str)

    n_predictions = int(len(df))
    n_bonds = int(df["cusip"].nunique())

    date_min = pd.to_datetime(df["date"].min())
    date_max = pd.to_datetime(df["date"].max())

    pred_desc = df["y_pred"].describe().to_dict()
    true_desc = df["y_true"].describe().to_dict()

    pred_range = float(df["y_pred"].max() - df["y_pred"].min())
    true_range = float(df["y_true"].max() - df["y_true"].min())
    range_ratio = float(pred_range / true_range) if true_range > 0 else float("nan")

    correlation = float(np.corrcoef(df["y_true"], df["y_pred"])[0, 1])

    residuals = df["y_true"] - df["y_pred"]
    mean_bias = float(residuals.mean())
    rmse = float(np.sqrt(np.mean(residuals.to_numpy() ** 2)))
    residual_std = float(residuals.std())

    df["year"] = df["date"].dt.year
    yearly_rmse = (
        df.groupby("year")
        .apply(lambda x: float(np.sqrt(np.mean((x["y_true"] - x["y_pred"]) ** 2))))
        .to_dict()
    )

    bond_rmse = df.groupby("cusip").apply(lambda x: float(np.sqrt(np.mean((x["y_true"] - x["y_pred"]) ** 2))))
    bond_rmse_desc = {
        "mean": float(bond_rmse.mean()),
        "std": float(bond_rmse.std()),
        "min": float(bond_rmse.min()),
        "max": float(bond_rmse.max()),
        "p25": float(bond_rmse.quantile(0.25)),
        "p50": float(bond_rmse.quantile(0.50)),
        "p75": float(bond_rmse.quantile(0.75)),
        "p95": float(bond_rmse.quantile(0.95)),
    }

    rmse_q75 = bond_rmse.quantile(0.75)
    rmse_q25 = bond_rmse.quantile(0.25)
    iqr = rmse_q75 - rmse_q25
    outlier_threshold = rmse_q75 + 1.5 * iqr
    outlier_bonds = bond_rmse[bond_rmse > outlier_threshold].nlargest(10).to_dict()

    prediction_uniqueness_ratio = float(df["y_pred"].nunique() / max(len(df), 1))

    abs_errors = residuals.abs()
    error_desc = {
        "mean_abs_error": float(abs_errors.mean()),
        "median_abs_error": float(abs_errors.median()),
        "p90_abs_error": float(abs_errors.quantile(0.90)),
        "p95_abs_error": float(abs_errors.quantile(0.95)),
        "p99_abs_error": float(abs_errors.quantile(0.99)),
    }
    fat_tail_ratio = float(abs_errors.quantile(0.95) / max(abs_errors.median(), 1e-12))

    issues: List[str] = []
    if range_ratio < 0.5:
        issues.append("Predictions are conservative relative to the observed range")
    if range_ratio > 1.5:
        issues.append("Predictions are overly volatile relative to the observed range")
    if correlation < 0.4:
        issues.append("Low correlation between predictions and true values")
    if abs(mean_bias) > rmse * 0.1:
        issues.append("Non-trivial mean bias in residuals")
    if prediction_uniqueness_ratio < 0.3:
        issues.append("Low diversity in predicted values (possible underfitting)")
    if fat_tail_ratio > 5.0:
        issues.append("Fat-tailed error distribution (tail risk / outliers)")
    return EnsembleDiagnostics(
        n_predictions=n_predictions,
        n_bonds=n_bonds,
        date_min=date_min,
        date_max=date_max,
        pred_desc={k: float(v) for k, v in pred_desc.items() if isinstance(v, (int, float, np.floating))},
        true_desc={k: float(v) for k, v in true_desc.items() if isinstance(v, (int, float, np.floating))},
        range_ratio=range_ratio,
        correlation=correlation,
        mean_bias=mean_bias,
        rmse=rmse,
        residual_std=residual_std,
        yearly_rmse={int(k): float(v) for k, v in yearly_rmse.items()},
        bond_rmse_desc=bond_rmse_desc,
        outlier_bonds={str(k): float(v) for k, v in outlier_bonds.items()},
        prediction_uniqueness_ratio=prediction_uniqueness_ratio,
        error_desc=error_desc,
        fat_tail_ratio_95_50=fat_tail_ratio,
        issues=tuple(issues),
    )
