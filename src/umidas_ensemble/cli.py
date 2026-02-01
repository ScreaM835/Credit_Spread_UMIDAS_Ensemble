from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from .config import EnsembleConfig, PathsConfig, UMIDASConfig
from .data import load_panel_data
from .logging_utils import get_logger
from .metrics import macro_over_bonds, micro_metrics_df, pct_improve, random_walk_baseline, save_metrics
from .plots import create_ensemble_visualizations
from .runner import run_advanced_ensemble_umidas

logger = get_logger(__name__)


def _parse_int_list(values: str) -> tuple[int, ...]:
    parts = [p.strip() for p in values.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def cmd_run(args: argparse.Namespace) -> int:
    paths = PathsConfig(
        data_path=Path(args.data),
        dict_path=Path(args.dict) if args.dict else None,
        save_dir=Path(args.save_dir),
        variable_list_cache=Path(args.variable_list_cache) if args.variable_list_cache else None,
    )
    umidas = UMIDASConfig(
        lm=args.lm,
        q_taps=args.q_taps,
        pub_lag=args.pub_lag,
        ar_lags=tuple(args.ar_lags),
        winsor_q=args.winsor_q,
        min_train_min=args.min_train_min,
        min_train_max=args.min_train_max,
    )
    ensemble = EnsembleConfig(
        horizons=_parse_int_list(args.horizons),
        meta_learning_method=args.meta,
        feature_engineering=args.feature_engineering,
        validation_months=args.validation_months,
        random_state=args.random_state,
        resume=not args.no_resume,
        log_every=args.log_every,
        n_threads=args.n_threads,
        n_jobs=args.n_jobs,
    )

    run_advanced_ensemble_umidas(paths=paths, umidas=umidas, ensemble=ensemble, fill_mode=args.fill_mode)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    pred = pd.read_csv(pred_path)
    pred["date"] = pd.to_datetime(pred["date"], errors="coerce")
    pred["cusip"] = pred["cusip"].astype(str)

    micro = micro_metrics_df(pred)
    macro, per_bond = macro_over_bonds(pred)

    save_dir = Path(args.save_dir) if args.save_dir else pred_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    save_metrics(save_dir=save_dir, horizon=args.horizon, macro=macro, per_bond=per_bond, tag="ensemble")

    logger.info("Micro RMSE: %.6f", micro["RMSE"])
    logger.info("Macro mean RMSE: %.6f", macro["macro_rmse_mean"])

    # Baseline evaluation (random walk / no-change): cs_{t-H} as predictor for cs_t
    if args.data:
        raw = load_panel_data(Path(args.data))
        baseline_df = random_walk_baseline(pred=pred, raw=raw, horizon=int(args.horizon))
        micro_b = micro_metrics_df(baseline_df)
        macro_b, per_bond_b = macro_over_bonds(baseline_df)

        save_metrics(save_dir=save_dir, horizon=args.horizon, macro=macro_b, per_bond=per_bond_b, tag="baseline")

        logger.info("Baseline micro RMSE: %.6f", micro_b["RMSE"])
        logger.info("RMSE improvement vs baseline: %.2f%%", pct_improve(micro["RMSE"], micro_b["RMSE"]))

    if args.plot:
        _fig, paths = create_ensemble_visualizations(pred=pred, save_path=save_dir / args.plot)
        logger.info("Saved figure to %s", paths.figure_path)

    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="umidas-ensemble")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run the walk-forward ensemble pipeline")
    run.add_argument("--data", required=True, help="Path to the panel CSV (must include cusip,date,cs)")
    run.add_argument("--dict", default=None, help="Path to the Excel variable dictionary (optional if cache exists)")
    run.add_argument("--save-dir", default="results", help="Output directory")
    run.add_argument("--variable-list-cache", default=None, help="Optional JSON cache path for variable lists")

    run.add_argument("--horizons", default="1", help="Comma-separated horizons in months (e.g. 1,3,6)")
    run.add_argument("--meta", default="stacking_ridge", help="Meta-learning method")
    run.add_argument("--feature-engineering", default="enhanced", help="Feature engineering mode")
    run.add_argument("--validation-months", type=int, default=6)
    run.add_argument("--random-state", type=int, default=42)
    run.add_argument("--no-resume", action="store_true", help="Disable resume and overwrite output files")
    run.add_argument("--log-every", type=int, default=1)
    run.add_argument("--n-threads", type=int, default=2)
    run.add_argument("--n-jobs", type=int, default=4)
    run.add_argument("--fill-mode", default="ffill_bfill", choices=["ffill_bfill", "ffill", "none"])

    # U-MIDAS config parameters
    run.add_argument("--lm", type=int, default=6)
    run.add_argument("--q-taps", type=int, default=4)
    run.add_argument("--pub-lag", type=int, default=1)
    run.add_argument("--ar-lags", type=int, nargs="+", default=[1, 2, 3, 6, 12])
    run.add_argument("--winsor-q", type=float, default=0.01)
    run.add_argument("--min-train-min", type=int, default=24)
    run.add_argument("--min-train-max", type=int, default=80)

    evalp = sub.add_parser("evaluate", help="Evaluate predictions and optionally plot diagnostics")
    evalp.add_argument("--predictions", required=True, help="Prediction CSV file produced by `run`")
    evalp.add_argument("--horizon", type=int, default=1, help="Forecast horizon in months")
    evalp.add_argument("--data", default=None, help="Panel CSV to build a baseline (optional)")
    evalp.add_argument("--save-dir", default=None, help="Output directory for metrics and plots")
    evalp.add_argument("--plot", default=None, help="If set, filename for saving a diagnostic figure")

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "evaluate":
        return cmd_evaluate(args)

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
