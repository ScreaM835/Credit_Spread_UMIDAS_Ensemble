from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import EnsembleConfig, PathsConfig, UMIDASConfig, VariableLists
from .data import build_variable_lists_from_excel, load_panel_data, load_variable_lists
from .features import UMIDASFeatureBuilder
from .logging_utils import get_logger
from .models import FinancialMetaLearner, engineer_features_optimized, get_base_learners, train_model_robust
from .utils import cpu_guard

logger = get_logger(__name__)


def _prepare_cs_series(df: pd.DataFrame) -> pd.Series:
    df_me = df.copy()
    df_me["_date_me"] = UMIDASFeatureBuilder.month_end_series(df_me["date"])
    df_me["cusip"] = df_me["cusip"].astype(str)
    cs_series = (
        df_me.set_index(["cusip", "_date_me"])["cs"]
        .astype(float)
        .sort_index()
    )
    return cs_series


def _all_month_ends(df: pd.DataFrame) -> np.ndarray:
    df_me = UMIDASFeatureBuilder.month_end_series(df["date"])
    return np.sort(df_me.dropna().unique())


def _compute_global_median(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].median(numeric_only=True)


def resolve_variable_lists(
    *,
    df: pd.DataFrame,
    paths: PathsConfig,
) -> VariableLists:
    """Resolve variable lists either from cache or from the Excel dictionary."""
    cache_path = paths.variable_list_cache
    if cache_path is None:
        cache_path = paths.save_dir / "u_midas_variable_lists.json"

    if cache_path.exists():
        monthly, quarterly = load_variable_lists(cache_path)
        logger.info("Loaded variable lists from %s", cache_path)
        return VariableLists.from_iterables(monthly, quarterly)

    if paths.dict_path is None:
        raise ValueError(
            "No variable list cache found and no Excel dictionary provided. "
            "Provide `dict_path` or `variable_list_cache`."
        )

    monthly, quarterly = build_variable_lists_from_excel(df=df, dict_path=paths.dict_path, cache_path=cache_path)
    return VariableLists.from_iterables(monthly, quarterly)


def run_advanced_ensemble_umidas(
    *,
    paths: PathsConfig,
    umidas: UMIDASConfig = UMIDASConfig(),
    ensemble: EnsembleConfig = EnsembleConfig(),
    fill_mode: str = "ffill_bfill",
) -> Dict[int, Path]:
    """Run the walk-forward Advanced Ensemble U-MIDAS pipeline.

    This is a package-quality refactor of the original notebook implementation.

    Outputs
    -------
    A dictionary mapping each horizon H to the path of a prediction CSV:
    - ``cusip``: bond identifier
    - ``date``: forecast target month-end (t)
    - ``y_true``: observed credit spread level cs_t
    - ``y_pred``: predicted credit spread level \hat{cs}_t
    - ``cs_t0``: observed cs_{t-H} used as the level baseline
    - ``y_delta_true``: cs_t - cs_{t-H}
    - ``y_delta_pred``: \hat{cs}_t - cs_{t-H}
    """
    paths.save_dir.mkdir(parents=True, exist_ok=True)

    df = load_panel_data(paths.data_path)
    variables = resolve_variable_lists(df=df, paths=paths)

    builder = UMIDASFeatureBuilder(df=df, variables=variables, config=umidas, fill_mode=fill_mode)

    cs_series = _prepare_cs_series(df)
    all_dates = _all_month_ends(df)

    numeric_cols = [c for c in (list(variables.monthly) + list(variables.quarterly) + ["cs"]) if c in df.columns]
    global_median = _compute_global_median(df, numeric_cols)

    logger.info(
        "Configured run: horizons=%s, meta=%s, feature_engineering=%s",
        ensemble.horizons,
        ensemble.meta_learning_method,
        ensemble.feature_engineering,
    )

    out_handles: Dict[int, Path] = {}

    # Precompute month-end aligned date and cusip arrays for fast bond selection.
    df_me = df.copy()
    df_me["_date_me"] = UMIDASFeatureBuilder.month_end_series(df_me["date"])
    df_me["cusip"] = df_me["cusip"].astype(str)
    df_dates_me = df_me["_date_me"]
    df_cusip = df_me["cusip"].to_numpy()

    for H in ensemble.horizons:
        preds_path = paths.save_dir / f"u_midas_advanced_ensemble_lead{H}_predictions.csv"

        completed = set()
        if ensemble.resume and preds_path.exists():
            try:
                past = pd.read_csv(preds_path, usecols=["date"])
                completed = set(builder.month_end_series(past["date"]).unique())
                logger.info("Resume enabled: %d completed dates detected for H=%s", len(completed), H)
            except Exception:
                completed = set()

        meta_learner = FinancialMetaLearner(method=ensemble.meta_learning_method)

        for i in range(umidas.min_train_max, len(all_dates) - 1):
            train_end = pd.Timestamp(all_dates[i])
            test_date = pd.Timestamp(all_dates[i + 1])

            if ensemble.resume and test_date in completed:
                continue

            pack = builder.pooled_design_up_to(train_end=train_end, horizon=H, global_median=global_median)
            if pack is None:
                continue

            X_raw, y, _groups_unused, _bonds_train, _dates_train = pack

            X_train = engineer_features_optimized(X_raw, ensemble.feature_engineering)

            n_samples = len(X_train)
            val_size = min(ensemble.validation_months * 50, n_samples // 4)

            if val_size > 20:
                X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
                y_tr, y_val = y[:-val_size], y[-val_size:]
            else:
                X_tr, X_val, y_tr, y_val = X_train, None, y, None

            with cpu_guard(ensemble.n_threads):
                from sklearn.preprocessing import RobustScaler

                scaler = RobustScaler()
                X_tr_scaled = scaler.fit_transform(X_tr)
                X_val_scaled = scaler.transform(X_val) if X_val is not None else None

            base_learners = get_base_learners(random_state=ensemble.random_state, n_jobs=ensemble.n_jobs)
            trained_models: dict[str, object] = {}

            for name, model in base_learners.items():
                trained_model, _status = train_model_robust(name, model, X_tr_scaled, y_tr, X_val_scaled, y_val)
                if trained_model is not None:
                    trained_models[name] = trained_model

            if not trained_models:
                logger.warning("No models trained for test date %s", test_date.date())
                continue

            # Meta-learning on validation predictions (holdout stacking)
            val_predictions: dict[str, np.ndarray] = {}
            if X_val_scaled is not None and y_val is not None:
                for name, model in trained_models.items():
                    try:
                        pred_val = model.predict(X_val_scaled)
                        if not (np.isnan(pred_val).any() or np.isinf(pred_val).any()):
                            val_predictions[name] = pred_val
                    except Exception:
                        continue

            if val_predictions and y_val is not None:
                meta_learner.fit(val_predictions, y_val, dates=None)

            # Build test design at t0 = test_date - H
            bonds_test = np.unique(df_cusip[df_dates_me == builder.month_end_scalar(test_date)])
            if bonds_test.size == 0:
                continue

            X_ref, bonds_ref = builder.t0_rows_for_bonds(
                bonds_req=bonds_test,
                t0_scalar=test_date - pd.DateOffset(months=int(H)),
                horizon=H,
                global_median=global_median,
            )

            if X_ref.shape[0] == 0:
                continue

            X_test = engineer_features_optimized(X_ref, ensemble.feature_engineering)
            X_test_scaled = scaler.transform(X_test)

            test_predictions: dict[str, np.ndarray] = {}
            for name, model in trained_models.items():
                try:
                    p = model.predict(X_test_scaled)
                    if not (np.isnan(p).any() or np.isinf(p).any()):
                        test_predictions[name] = p
                except Exception:
                    continue

            if not test_predictions:
                continue

            try:
                y_delta_pred = meta_learner.predict(test_predictions)
            except Exception:
                y_delta_pred = np.mean(list(test_predictions.values()), axis=0)

            # Align true cs values
            t0_me = builder.month_end_scalar(test_date - pd.DateOffset(months=int(H)))
            t_me = builder.month_end_scalar(test_date)

            pairs_t0 = pd.MultiIndex.from_arrays([bonds_ref.astype(str), np.full(bonds_ref.shape, t0_me)])
            pairs_t = pd.MultiIndex.from_arrays([bonds_ref.astype(str), np.full(bonds_ref.shape, t_me)])

            cs_t0 = cs_series.reindex(pairs_t0).to_numpy()
            cs_t = cs_series.reindex(pairs_t).to_numpy()

            ok = np.isfinite(cs_t0) & np.isfinite(cs_t) & np.isfinite(y_delta_pred)
            if not np.any(ok):
                continue

            y_pred = cs_t0[ok] + y_delta_pred[ok]
            y_true = cs_t[ok]
            y_delta_true = y_true - cs_t0[ok]

            out_df = pd.DataFrame(
                {
                    "cusip": bonds_ref[ok].astype(str),
                    "date": t_me,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "cs_t0": cs_t0[ok],
                    "y_delta_true": y_delta_true,
                    "y_delta_pred": y_pred - cs_t0[ok],
                }
            )

            write_header = not preds_path.exists()
            out_df.to_csv(
                preds_path,
                mode=("w" if write_header else "a"),
                header=write_header,
                index=False,
            )

            if (i - umidas.min_train_max) % max(ensemble.log_every, 1) == 0:
                logger.info(
                    "train_end=%s -> test_date=%s | n_bonds=%d | n_models=%d",
                    train_end.date(),
                    test_date.date(),
                    len(out_df),
                    len(trained_models),
                )

            # Cleanup
            del trained_models, test_predictions, X_train, X_tr_scaled
            if X_val_scaled is not None:
                del X_val_scaled
            gc.collect()

        # Summary
        try:
            fin = pd.read_csv(preds_path)
            n_bonds = fin["cusip"].nunique() if "cusip" in fin.columns else 0
            logger.info("Completed H=%s: %d rows, %d bonds -> %s", H, len(fin), n_bonds, preds_path)
        except Exception:
            logger.warning("Completed H=%s but could not load prediction summary.", H)

        out_handles[int(H)] = preds_path

    return out_handles
