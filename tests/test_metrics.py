import numpy as np
import pandas as pd

from umidas_ensemble.metrics import micro_metrics_df


def test_micro_metrics_perfect_prediction():
    df = pd.DataFrame(
        {
            "y_true": [1.0, 2.0, 3.0],
            "y_pred": [1.0, 2.0, 3.0],
        }
    )
    m = micro_metrics_df(df)
    assert m["RMSE"] == 0.0
    assert m["MAE"] == 0.0
    assert abs(m["R2"] - 1.0) < 1e-12


def test_micro_metrics_simple_error():
    df = pd.DataFrame(
        {
            "y_true": [1.0, 2.0],
            "y_pred": [0.0, 2.0],
        }
    )
    m = micro_metrics_df(df)
    # Errors are [1, 0] => MSE=0.5, RMSE=sqrt(0.5)
    assert abs(m["MSE"] - 0.5) < 1e-12
    assert abs(m["RMSE"] - np.sqrt(0.5)) < 1e-12
