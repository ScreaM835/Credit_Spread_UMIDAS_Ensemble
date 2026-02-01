from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import RobustScaler

from .logging_utils import get_logger

logger = get_logger(__name__)


def _try_import_lightgbm():
    try:
        import lightgbm as lgb  # type: ignore

        return lgb
    except Exception:
        return None


def _try_import_xgboost():
    try:
        import xgboost as xgb  # type: ignore

        return xgb
    except Exception:
        return None


def get_base_learners(random_state: int = 42, n_jobs: int = 4) -> Dict[str, object]:
    """Return the base learners used in the ensemble.

    The selection mirrors the notebook:
    - ElasticNet
    - Ridge
    - Gradient boosting (LightGBM if available; otherwise XGBoost; otherwise a safe fallback)
    - Random Forest (shallow, constrained)

    Notes
    -----
    If neither LightGBM nor XGBoost are installed, a sklearn GradientBoostingRegressor
    fallback is used. This deviates from the notebook but preserves functionality.
    """
    lgb = _try_import_lightgbm()
    xgb = _try_import_xgboost()

    # 1) ElasticNet
    elasticnet = ElasticNet(
        alpha=0.01,
        l1_ratio=0.5,
        max_iter=3000,
        tol=1e-4,
        random_state=random_state,
    )

    # 2) Ridge
    ridge = Ridge(
        alpha=0.1,
        solver="auto",
        random_state=random_state,
    )

    # 3) Boosting model
    if lgb is not None:
        gbm = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=-1,
            force_row_wise=True,
        )
    elif xgb is not None:
        gbm = xgb.XGBRegressor(
            objective="reg:absoluteerror",
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=random_state,
            n_jobs=n_jobs,
            tree_method="hist",
            verbosity=0,
        )
    else:
        from sklearn.ensemble import GradientBoostingRegressor

        warnings.warn(
            "Neither LightGBM nor XGBoost is available. Falling back to sklearn GradientBoostingRegressor.",
            RuntimeWarning,
        )
        gbm = GradientBoostingRegressor(
            random_state=random_state,
        )

    # 4) Random Forest
    rf = RandomForestRegressor(
        n_estimators=80,
        max_depth=4,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0,
    )

    return {
        "elasticnet_main": elasticnet,
        "ridge_robust": ridge,
        "gbm_fast": gbm,
        "rf_fast": rf,
    }


class FinancialMetaLearner:
    """Stacking-style meta-learner with optional regime awareness.

    The meta-learner is trained on *base model predictions* as meta-features.
    For the default method (``stacking_ridge``), coefficients are constrained to be
    non-negative, encouraging an interpretable weighted combination.

    Parameters
    ----------
    method:
        One of: stacking_ridge, stacking_elastic, neural_meta, regime_aware.
    window_size:
        Reserved for future use.
    """

    def __init__(self, method: str = "stacking_ridge", window_size: int = 252):
        self.method = method
        self.window_size = window_size
        self.meta_model = None
        self.regime_model = None
        self.regime_meta_models: dict[int, object] = {}

    def _get_meta_model(self):
        if self.method == "stacking_ridge":
            return Ridge(alpha=0.1, positive=True, random_state=42)
        if self.method == "stacking_elastic":
            return ElasticNet(alpha=0.01, l1_ratio=0.5, positive=True, random_state=42)
        if self.method == "neural_meta":
            try:
                from sklearn.neural_network import MLPRegressor

                return MLPRegressor(
                    hidden_layer_sizes=(32, 16),
                    alpha=0.01,
                    max_iter=1000,
                    random_state=42,
                )
            except Exception:
                return Ridge(alpha=0.1, positive=True, random_state=42)
        # regime_aware falls back to ridge for each regime
        return Ridge(alpha=0.1, positive=True, random_state=42)

    def fit(self, base_predictions: dict[str, np.ndarray], y_true: np.ndarray, dates=None):
        """Fit the meta-learner."""
        if len(base_predictions) == 0:
            return self

        X_meta = np.column_stack(list(base_predictions.values()))

        if self.method == "regime_aware" and dates is not None:
            try:
                from sklearn.mixture import GaussianMixture

                mean_pred = np.mean(X_meta, axis=1)
                residuals = y_true - mean_pred
                volatility = pd.Series(residuals).rolling(window=12, min_periods=3).std().fillna(0)

                regime_features = np.column_stack([residuals.reshape(-1, 1), volatility.values.reshape(-1, 1)])

                self.regime_model = GaussianMixture(n_components=3, random_state=42)
                regimes = self.regime_model.fit_predict(regime_features)

                for regime in np.unique(regimes):
                    regime_mask = regimes == regime
                    if np.sum(regime_mask) > 10:
                        model = self._get_meta_model()
                        model.fit(X_meta[regime_mask], y_true[regime_mask])
                        self.regime_meta_models[int(regime)] = model

                if not self.regime_meta_models:
                    self.meta_model = self._get_meta_model()
                    self.meta_model.fit(X_meta, y_true)

            except Exception as e:
                logger.warning("Regime-aware training failed (%s); falling back to stacking_ridge.", e)
                self.method = "stacking_ridge"
                self.meta_model = self._get_meta_model()
                self.meta_model.fit(X_meta, y_true)
        else:
            self.meta_model = self._get_meta_model()
            self.meta_model.fit(X_meta, y_true)

        return self

    def predict(self, base_predictions: dict[str, np.ndarray], current_state=None) -> np.ndarray:
        """Predict using the meta-learner."""
        if len(base_predictions) == 0:
            return np.array([], dtype=float)

        X_meta = np.column_stack(list(base_predictions.values()))

        if self.method == "regime_aware" and self.regime_model is not None and self.regime_meta_models:
            try:
                if current_state is not None:
                    regime_features = np.array(current_state).reshape(1, -1)
                    regime = int(self.regime_model.predict(regime_features)[0])
                    if regime in self.regime_meta_models:
                        return self.regime_meta_models[regime].predict(X_meta)

                # fallback: average over regime models
                preds = [m.predict(X_meta) for m in self.regime_meta_models.values()]
                if preds:
                    return np.mean(preds, axis=0)
            except Exception as e:
                logger.warning("Regime prediction failed (%s); using standard model.", e)

        if self.meta_model is not None:
            return self.meta_model.predict(X_meta)

        return np.mean(X_meta, axis=1)


def engineer_features_optimized(X_raw: np.ndarray, feature_mode: str = "basic") -> np.ndarray:
    """Feature engineering from the notebook.

    The default ``basic`` returns the raw design matrix (U-MIDAS features).

    The ``enhanced`` mode implements a small number of additional heuristics:
    - differences across the first few AR columns ("momentum");
    - a short rolling standard deviation computed across *rows*;
    - limited pairwise interactions among the most variant features.

    Notes
    -----
    The enhanced features are heuristic. In pooled panel settings, row-order may
    mix issuers and calendar times. Users seeking a strictly time-ordered
    transformation should pre-sort the pooled design or implement a date-aware
    transformation.
    """
    X = X_raw.copy().astype(np.float64)

    if feature_mode == "basic":
        return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    n_samples, n_features = X.shape
    engineered: list[np.ndarray] = []

    # 1) AR momentum
    ar_cols = min(5, n_features)
    if ar_cols > 1:
        ar_data = X[:, :ar_cols]
        ar_momentum = np.diff(ar_data, axis=1, prepend=ar_data[:, 0:1])
        engineered.append(ar_momentum)

    # 2) Volatility proxy
    if n_samples > 3:
        rolling_std = np.zeros_like(X)
        for i in range(3, n_samples):
            window = X[max(0, i - 3) : i + 1]
            rolling_std[i] = np.std(window, axis=0, ddof=0)

        vol_importance = np.mean(rolling_std, axis=0)
        top_vol_idx = np.argsort(vol_importance)[-5:]
        engineered.append(rolling_std[:, top_vol_idx])

    # 3) Limited interactions
    if n_features <= 30:
        feature_vars = np.var(X, axis=0)
        top_features = np.argsort(feature_vars)[-min(6, max(1, n_features // 2)) :]

        interaction_count = 0
        for i in range(len(top_features)):
            for j in range(i + 1, len(top_features)):
                if interaction_count < 10:
                    interaction = (X[:, top_features[i]] * X[:, top_features[j]]).reshape(-1, 1)
                    engineered.append(interaction)
                    interaction_count += 1

    if engineered:
        try:
            valid = [f for f in engineered if f.shape[0] == n_samples]
            X_enhanced = np.hstack([X] + valid) if valid else X
        except Exception:
            X_enhanced = X
    else:
        X_enhanced = X

    return np.nan_to_num(X_enhanced, nan=0.0, posinf=1e6, neginf=-1e6)


def train_model_robust(
    name: str,
    model: object,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> Tuple[Optional[object], str]:
    """Train a model with a simple fallback hierarchy."""
    try:
        model.fit(X_train, y_train)

        if X_val is not None and y_val is not None:
            y_pred = model.predict(X_val)
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                raise ValueError("Invalid predictions")

        return model, "success"

    except Exception as e:
        logger.warning("Model '%s' failed to fit (%s). Applying fallbacks.", name, e)

        fallback_models = [
            ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=42)
            if "elastic" in name.lower()
            else None,
            Ridge(alpha=0.1, random_state=42),
            Ridge(alpha=1.0, random_state=42),
        ]

        for fb in fallback_models:
            if fb is None:
                continue
            try:
                fb.fit(X_train, y_train)
                return fb, f"fallback_{type(fb).__name__}"
            except Exception:
                continue

        return None, "failed"


@dataclass(frozen=True)
class ScalerBundle:
    scaler: RobustScaler

    @classmethod
    def fit(cls, X: np.ndarray) -> "ScalerBundle":
        scaler = RobustScaler()
        scaler.fit(X)
        return cls(scaler=scaler)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)
