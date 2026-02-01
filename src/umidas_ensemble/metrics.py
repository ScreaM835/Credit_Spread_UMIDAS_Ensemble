from __future__ import annotations

import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .features import UMIDASFeatureBuilder

EPS = 1e-12


def _to_1d(a) -> np.ndarray:
    if isinstance(a, pd.DataFrame):
        return a.iloc[:, 0].to_numpy(float)
    return np.asarray(a, float).reshape(-1)


def micro_metrics_df(df: pd.DataFrame) -> Dict[str, float]:
    """Compute micro-averaged metrics across all rows."""
    y = _to_1d(df["y_true"])
    yp = _to_1d(df["y_pred"])
    e = y - yp

    mse = float(np.mean(e**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(e)))

    sst = float(np.sum((y - np.mean(y)) ** 2))
    sse = float(np.sum(e**2))
    r2 = 1.0 - sse / max(sst, EPS)

    den = np.maximum(np.abs(y), EPS)
    pae = float(np.mean(100.0 * np.abs(e) / den))
    mape = pae
    mpe = float(np.mean(100.0 * e / den))
    smape = float(np.mean(100.0 * 2.0 * np.abs(e) / (np.abs(y) + np.abs(yp) + EPS)))
    rrmse = float(100.0 * rmse / (np.mean(np.abs(y)) + EPS))

    return dict(
        MSE=mse,
        RMSE=rmse,
        MAE=mae,
        R2=r2,
        PAE_pct=pae,
        MAPE_pct=mape,
        MPE_pct=mpe,
        sMAPE_pct=smape,
        rRMSE_pct=rrmse,
    )


def per_bond_metrics(df: pd.DataFrame) -> pd.Series:
    m = micro_metrics_df(df)
    m["cusip"] = df["cusip"].iloc[0]
    m["n"] = int(len(df))
    return pd.Series(m)


def macro_over_bonds(df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Compute macro-averaged metrics across bonds.

    Macro averaging is performed by:
    1) computing per-bond micro metrics, then
    2) averaging those metrics across bonds.
    """
    required_cols = {"cusip", "y_true", "y_pred"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    bt = df.groupby("cusip", sort=False).apply(per_bond_metrics).reset_index(drop=True)

    macro = dict(
        macro_mse=float(bt["MSE"].mean()),
        macro_rmse_mean=float(bt["RMSE"].mean()),
        macro_rmse_sqrt=float(np.sqrt(bt["MSE"].mean())),
        macro_mae=float(bt["MAE"].mean()),
        macro_r2_mean=float(bt["R2"].mean()),
        macro_pae_pct=float(bt["PAE_pct"].mean()),
        macro_mape_pct=float(bt["MAPE_pct"].mean()),
        macro_mpe_pct=float(bt["MPE_pct"].mean()),
        macro_smape_pct=float(bt["sMAPE_pct"].mean()),
        macro_rrmse_pct=float(bt["rRMSE_pct"].mean()),
        n_bonds=int(bt.shape[0]),
        n_rows=int(df.shape[0]),
    )
    return macro, bt


def pct_improve(model_err: float, baseline_err: float) -> float:
    return 100.0 * (baseline_err - model_err) / max(baseline_err, EPS)


def random_walk_baseline(
    *,
    pred: pd.DataFrame,
    raw: pd.DataFrame,
    horizon: int,
    date_col: str = "date",
    cs_col: str = "cs",
) -> pd.DataFrame:
    """Construct a random-walk / no-change baseline aligned with the horizon.

    For each (cusip, date=t), baseline prediction is cs_{t-H}.

    Parameters
    ----------
    pred:
        Model prediction DataFrame with columns (cusip, date, y_true, y_pred).
    raw:
        Raw panel data with columns (cusip, date, cs).
    horizon:
        Forecast horizon in months.
    """
    if not {"cusip", date_col, "y_true"}.issubset(pred.columns):
        raise ValueError("pred must contain columns: cusip, date, y_true")
    if not {"cusip", date_col, cs_col}.issubset(raw.columns):
        raise ValueError("raw must contain columns: cusip, date, cs")

    p = pred.copy()
    p[date_col] = pd.to_datetime(p[date_col], errors="coerce")
    p["cusip"] = p["cusip"].astype(str)

    r = raw.copy()
    r[date_col] = pd.to_datetime(r[date_col], errors="coerce")
    r["cusip"] = r["cusip"].astype(str)

    # Normalise dates to month-ends.
    p["_t_me"] = UMIDASFeatureBuilder.month_end_series(p[date_col])
    r["_t_me"] = UMIDASFeatureBuilder.month_end_series(r[date_col])

    p["_t0_me"] = (p["_t_me"] - pd.DateOffset(months=int(horizon))).dt.to_period("M").dt.to_timestamp("M")

    r0 = r[["cusip", "_t_me", cs_col]].rename(columns={"_t_me": "_t0_me", cs_col: "y_pred"})
    out = p[["cusip", "y_true", "_t0_me"]].merge(r0, on=["cusip", "_t0_me"], how="left").dropna()

    return out[["cusip", "y_true", "y_pred"]]


@dataclass(frozen=True)
class HorizonSummary:
    horizon: int
    micro: Dict[str, float]
    macro: Dict[str, float]
    baseline_micro: Dict[str, float] | None = None
    baseline_macro: Dict[str, float] | None = None

    def to_dict(self) -> Dict[str, object]:
        d: Dict[str, object] = {"H": self.horizon, "micro": self.micro, "macro": self.macro}
        if self.baseline_micro is not None:
            d["baseline_micro"] = self.baseline_micro
        if self.baseline_macro is not None:
            d["baseline_macro"] = self.baseline_macro
        return d


def save_metrics(
    *,
    save_dir: Path,
    horizon: int,
    macro: Dict[str, float],
    per_bond: pd.DataFrame,
    tag: str,
) -> Tuple[Path, Path]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    macro_path = save_dir / f"u_midas_macro_metrics_H{horizon}_{tag}.json"
    per_bond_path = save_dir / f"u_midas_per_bond_metrics_H{horizon}_{tag}.csv"

    macro_path.write_text(json.dumps(macro, indent=2))
    per_bond.to_csv(per_bond_path, index=False)

    return macro_path, per_bond_path
