from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats


@dataclass(frozen=True)
class PlotPaths:
    figure_path: Optional[Path]


def create_ensemble_visualizations(
    *,
    pred: pd.DataFrame,
    save_path: Path | None = None,
    random_state: int = 42,
    sample_size: int = 10_000,
) -> Tuple[plt.Figure, PlotPaths]:
    """Create a 3x3 diagnostic figure for ensemble predictions."""
    df = pred.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["residuals"] = df["y_true"] - df["y_pred"]
    df["abs_residuals"] = df["residuals"].abs()

    rng = np.random.default_rng(random_state)

    fig = plt.figure(figsize=(20, 16))

    # 1) Prediction vs True scatter
    ax1 = plt.subplot(3, 3, 1)
    n = len(df)
    s = min(sample_size, n)
    idx = rng.choice(n, s, replace=False) if n > 0 else np.array([], dtype=int)
    sample = df.iloc[idx] if len(idx) else df

    ax1.scatter(sample["y_true"], sample["y_pred"], alpha=0.5, s=1)
    min_val = float(min(sample["y_true"].min(), sample["y_pred"].min()))
    max_val = float(max(sample["y_true"].max(), sample["y_pred"].max()))
    ax1.plot([min_val, max_val], [min_val, max_val], linewidth=2, label="Perfect prediction")
    ax1.set_xlabel("True values")
    ax1.set_ylabel("Predicted values")
    ax1.set_title("Predictions vs true values\n(sample)")
    ax1.legend()

    corr = float(np.corrcoef(df["y_true"], df["y_pred"])[0, 1]) if len(df) else float("nan")
    ax1.text(
        0.05,
        0.95,
        f"Correlation: {corr:.4f}",
        transform=ax1.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # 2) Residual distribution
    ax2 = plt.subplot(3, 3, 2)
    ax2.hist(df["residuals"], bins=100, alpha=0.7, density=True)
    ax2.axvline(df["residuals"].mean(), linestyle="--", label=f"Mean: {df['residuals'].mean():.6f}")
    ax2.set_xlabel("Residuals (true - predicted)")
    ax2.set_ylabel("Density")
    ax2.set_title("Residuals distribution")
    ax2.legend()

    # 3) Temporal RMSE
    ax3 = plt.subplot(3, 3, 3)
    df["year_month"] = df["date"].dt.to_period("M")
    monthly_rmse = df.groupby("year_month").apply(lambda x: float(np.sqrt(np.mean(x["residuals"] ** 2))))
    monthly_rmse.index = monthly_rmse.index.to_timestamp()
    ax3.plot(monthly_rmse.index, monthly_rmse.values, linewidth=2)
    ax3.set_xlabel("Date")
    ax3.set_ylabel("RMSE")
    ax3.set_title("Model performance over time")
    for tick in ax3.get_xticklabels():
        tick.set_rotation(45)

    # 4) Error by prediction magnitude
    ax4 = plt.subplot(3, 3, 4)
    df["pred_bins"] = pd.cut(df["y_pred"], bins=20, labels=False)
    err_by_mag = df.groupby("pred_bins").agg(abs_residuals=("abs_residuals", "mean"), y_pred=("y_pred", "mean"))
    ax4.scatter(err_by_mag["y_pred"], err_by_mag["abs_residuals"])
    ax4.set_xlabel("Prediction magnitude")
    ax4.set_ylabel("Mean absolute error")
    ax4.set_title("Error vs prediction magnitude")

    # 5) Top outlier bonds
    ax5 = plt.subplot(3, 3, 5)
    bond_rmse = df.groupby("cusip").apply(lambda x: float(np.sqrt(np.mean(x["residuals"] ** 2))))
    top = bond_rmse.nlargest(10)
    ax5.barh(range(len(top)), top.values)
    ax5.set_yticks(range(len(top)))
    ax5.set_yticklabels([f"{c[:8]}..." for c in top.index])
    ax5.set_xlabel("RMSE")
    ax5.set_title("Top 10 most challenging bonds")

    # 6) Error distribution by year
    ax6 = plt.subplot(3, 3, 6)
    df["year"] = df["date"].dt.year
    years = sorted(df["year"].unique())
    error_data = [df.loc[df["year"] == y, "abs_residuals"].values for y in years]
    ax6.boxplot(error_data, tick_labels=years)
    ax6.set_xlabel("Year")
    ax6.set_ylabel("Absolute error")
    ax6.set_title("Error distribution by year")
    for tick in ax6.get_xticklabels():
        tick.set_rotation(45)

    # 7) Q-Q plot
    ax7 = plt.subplot(3, 3, 7)
    qq_sample = df["residuals"].sample(min(10_000, len(df)), random_state=random_state)
    stats.probplot(qq_sample, dist="norm", plot=ax7)
    ax7.set_title("Q-Q plot: residuals vs normal")

    # 8) Range prediction accuracy (per bond)
    ax8 = plt.subplot(3, 3, 8)
    bond_stats = df.groupby("cusip").agg(true_min=("y_true", "min"), true_max=("y_true", "max"), pred_min=("y_pred", "min"), pred_max=("y_pred", "max"))
    bond_stats["true_range"] = bond_stats["true_max"] - bond_stats["true_min"]
    bond_stats["pred_range"] = bond_stats["pred_max"] - bond_stats["pred_min"]
    sample_bonds = bond_stats.sample(min(1000, len(bond_stats)), random_state=random_state)
    ax8.scatter(sample_bonds["true_range"], sample_bonds["pred_range"], alpha=0.6)
    max_range = float(max(sample_bonds["true_range"].max(), sample_bonds["pred_range"].max()))
    ax8.plot([0, max_range], [0, max_range], linewidth=2, label="Perfect range prediction")
    ax8.set_xlabel("True value range (per bond)")
    ax8.set_ylabel("Predicted value range (per bond)")
    ax8.set_title("Range prediction accuracy")
    ax8.legend()

    # 9) Cumulative error distribution
    ax9 = plt.subplot(3, 3, 9)
    sorted_errors = np.sort(df["abs_residuals"].to_numpy())
    cumulative_pct = np.arange(1, len(sorted_errors) + 1) / max(len(sorted_errors), 1) * 100
    ax9.plot(sorted_errors, cumulative_pct)
    ax9.set_xlabel("Absolute error")
    ax9.set_ylabel("Cumulative percentage")
    ax9.set_title("Cumulative error distribution")
    ax9.grid(True, alpha=0.3)

    p90, p95, p99 = np.percentile(sorted_errors, [90, 95, 99])
    ax9.axvline(p90, linestyle="--", label=f"90th: {p90:.6f}")
    ax9.axvline(p95, linestyle="--", label=f"95th: {p95:.6f}")
    ax9.axvline(p99, linestyle="--", label=f"99th: {p99:.6f}")
    ax9.legend()

    plt.tight_layout()

    fig_path = None
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        fig_path = save_path

    return fig, PlotPaths(figure_path=fig_path)


def performance_by_credit_spread_level(pred: pd.DataFrame) -> pd.DataFrame:
    """Summarise RMSE conditional on credit spread levels."""
    df = pred.copy()
    df["residuals"] = df["y_true"] - df["y_pred"]
    df["cs_level"] = pd.cut(
        df["y_true"],
        bins=[0, 0.01, 0.02, 0.05, 0.1, 1.0],
        labels=[
            "Very low (<1%)",
            "Low (1-2%)",
            "Medium (2-5%)",
            "High (5-10%)",
            "Very high (>10%)",
        ],
    )

    out = (
        df.groupby("cs_level")
        .agg(RMSE=("residuals", lambda x: float(np.sqrt(np.mean(x**2)))), Count=("residuals", "size"))
        .reset_index()
    )
    return out
