"""Synthetic demo for the Advanced Ensemble U-MIDAS package.

This example generates a small synthetic panel, writes it to disk, and runs a short
walk-forward forecast.

The objective is to demonstrate the API wiring; it is not intended to replicate the
research dataset or achieve strong predictive performance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from umidas_ensemble.config import EnsembleConfig, PathsConfig, UMIDASConfig, VariableLists
from umidas_ensemble.runner import run_advanced_ensemble_umidas


def make_synthetic_panel(n_bonds: int = 25, n_months: int = 36, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-31", periods=n_months, freq="M")

    rows = []
    for b in range(n_bonds):
        cusip = f"SYN{b:05d}"
        cs = np.zeros(n_months)
        x_m1 = rng.normal(size=n_months)
        x_m2 = rng.normal(size=n_months)
        x_q1 = rng.normal(size=n_months)

        # Simple AR(1)-like dynamics plus covariate influence
        for t in range(1, n_months):
            cs[t] = 0.9 * cs[t - 1] + 0.02 * x_m1[t - 1] - 0.01 * x_m2[t - 1] + 0.1 * rng.normal()

        cs = np.clip(0.02 + 0.01 * cs, 0.0001, 0.25)

        for t, dt in enumerate(dates):
            rows.append(
                {
                    "cusip": cusip,
                    "date": dt,
                    "cs": float(cs[t]),
                    "m_var1": float(x_m1[t]),
                    "m_var2": float(x_m2[t]),
                    "q_var1": float(x_q1[t]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    outdir = Path("synthetic_results")
    outdir.mkdir(exist_ok=True)

    df = make_synthetic_panel()
    data_path = outdir / "synthetic_panel.csv"
    df.to_csv(data_path, index=False)

    # Provide variable lists directly (bypassing the Excel dictionary).
    # In a real use case you would supply an Excel dictionary or a JSON cache.
    variables = VariableLists.from_iterables(monthly=["m_var1", "m_var2"], quarterly=["q_var1"])
    cache_path = outdir / "u_midas_variable_lists.json"
    import json
    cache_payload = {
        "monthly_vars_all": list(variables.monthly),
        "quarterly_vars_all": list(variables.quarterly),
        "counts": {"monthly": len(variables.monthly), "quarterly": len(variables.quarterly)},
    }
    cache_path.write_text(json.dumps(cache_payload, indent=2))

    paths = PathsConfig(data_path=data_path, dict_path=None, save_dir=outdir, variable_list_cache=cache_path)

    umidas = UMIDASConfig(min_train_min=6, min_train_max=12)
    ensemble = EnsembleConfig(horizons=(1,), validation_months=3, log_every=2, resume=False)

    run_advanced_ensemble_umidas(paths=paths, umidas=umidas, ensemble=ensemble)


if __name__ == "__main__":
    main()
