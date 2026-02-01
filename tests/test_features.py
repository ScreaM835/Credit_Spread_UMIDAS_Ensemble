import pandas as pd
import numpy as np

from umidas_ensemble.config import UMIDASConfig, VariableLists
from umidas_ensemble.features import UMIDASFeatureBuilder


def test_design_matrix_shapes():
    dates = pd.date_range("2020-01-31", periods=24, freq="M")
    df = pd.DataFrame(
        {
            "cusip": ["A"] * len(dates),
            "date": dates,
            "cs": np.linspace(0.01, 0.02, len(dates)),
            "m1": np.linspace(0.0, 1.0, len(dates)),
            "q1": np.linspace(0.0, 1.0, len(dates)),
        }
    )
    variables = VariableLists.from_iterables(monthly=["m1"], quarterly=["q1"])
    cfg = UMIDASConfig(lm=6, q_taps=4, pub_lag=1, ar_lags=(1, 2, 3, 6, 12), min_train_min=6, min_train_max=12)
    builder = UMIDASFeatureBuilder(df=df, variables=variables, config=cfg)

    global_median = df[["m1", "q1", "cs"]].median(numeric_only=True)

    dm = builder.build_design_for_bond(df, horizon=1, global_median=global_median)

    # Features: 5 AR + 6 monthly taps + 4 quarterly taps = 15
    assert dm.X.shape == (len(dates), 15)
    assert dm.y_delta.shape == (len(dates),)
