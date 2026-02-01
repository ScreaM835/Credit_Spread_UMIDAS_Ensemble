from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .config import UMIDASConfig, VariableLists


def winsorize_inplace(arr: np.ndarray, q: float) -> None:
    """Winsorise an array in-place using lower/upper quantiles."""
    if arr.size == 0:
        return
    lo, hi = np.nanpercentile(arr, [100 * q, 100 * (1 - q)])
    np.clip(arr, lo, hi, out=arr)


def fill_series(
    x: np.ndarray,
    fallback: float,
    *,
    mode: str = "ffill_bfill",
) -> np.ndarray:
    """Impute missing values in a 1D series.

    This mirrors the original notebook behaviour by default (forward fill followed
    by backward fill, then fallback).

    Parameters
    ----------
    x:
        Input array.
    fallback:
        Value used if all entries are missing after fills.
    mode:
        - ``"ffill_bfill"``: forward-fill then backward-fill (not strictly causal).
        - ``"ffill"``: forward-fill only (causal).
        - ``"none"``: no fill; NaNs are replaced with fallback.

    Notes
    -----
    The default ``ffill_bfill`` may introduce look-ahead if the series contains
    internal gaps. For strictly causal experiments, use ``mode="ffill"``.
    """
    y = x.astype(np.float32, copy=True)

    if mode not in {"ffill_bfill", "ffill", "none"}:
        raise ValueError(f"Unknown fill mode: {mode}")

    if mode in {"ffill_bfill", "ffill"}:
        n = len(y)
        last = np.nan
        for i in range(n):
            if np.isfinite(y[i]):
                last = y[i]
            else:
                y[i] = last

    if mode == "ffill_bfill":
        n = len(y)
        last = np.nan
        for i in range(n - 1, -1, -1):
            if np.isfinite(y[i]):
                last = y[i]
            else:
                y[i] = last

    y[~np.isfinite(y)] = fallback
    return y


def monthly_windows(x: np.ndarray, L: int) -> np.ndarray:
    """Construct monthly lag windows using a one-period shift.

    For each time t, returns [x_{t-1}, x_{t-2}, ..., x_{t-L}].

    The shift by 1 month avoids using contemporaneous x_t.
    """
    z = np.empty_like(x)
    z[0] = np.nan
    z[1:] = x[:-1]
    if len(z) < L:
        return np.full((len(z), L), np.nan, dtype=np.float32)
    w = sliding_window_view(z, L)
    out = np.full((len(z), L), np.nan, dtype=np.float32)
    out[L - 1 :, :] = w
    return out


def quarterly_taps(x: np.ndarray, pub_lag: int, Q: int) -> np.ndarray:
    """Construct quarterly taps spaced 3 months apart with a publication lag."""
    n = len(x)
    out = np.full((n, Q), np.nan, dtype=np.float32)
    for t in range(n):
        ok = True
        vals = []
        for q in range(Q):
            lag = pub_lag + 3 * q
            idx = t - lag
            if idx < 0:
                ok = False
                break
            vals.append(x[idx])
        if ok:
            out[t, :] = np.array(vals, dtype=np.float32)
    return out


def ar_lags(cs: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    """Autoregressive lags of the target level series."""
    T = len(cs)
    A = np.full((T, len(lags)), np.nan, dtype=np.float32)
    for j, L in enumerate(lags):
        if L < T:
            A[L:, j] = cs[:-L]
    return A


def _stable_int64(label: str) -> np.int64:
    """Deterministic 64-bit hash for feature group identifiers."""
    h = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    return np.int64(int.from_bytes(h, byteorder="little", signed=True))


@dataclass(frozen=True)
class DesignMatrix:
    X: np.ndarray
    y_delta: np.ndarray
    dates: np.ndarray  # datetime64
    mask: np.ndarray   # bool
    cs_level: np.ndarray
    group_ids: np.ndarray


class UMIDASFeatureBuilder:
    """U-MIDAS design matrix builder for a panel dataset."""

    def __init__(
        self,
        *,
        df: pd.DataFrame,
        variables: VariableLists,
        config: UMIDASConfig = UMIDASConfig(),
        fill_mode: str = "ffill_bfill",
    ) -> None:
        self.df = df
        self.variables = variables
        self.config = config
        self.fill_mode = fill_mode

    def build_design_for_bond(
        self,
        g: pd.DataFrame,
        *,
        horizon: int,
        global_median: pd.Series,
    ) -> DesignMatrix:
        """Build per-bond design matrix for a given horizon."""
        cfg = self.config
        L = cfg.lm
        Q = cfg.q_taps
        pub_lag = cfg.pub_lag

        T = len(g)
        cs = g["cs"].to_numpy(np.float32)

        H = int(horizon)
        cs_leadH = np.roll(cs, -H)
        cs_leadH[-H:] = np.nan
        y_delta = cs_leadH - cs

        blocks: list[np.ndarray] = [ar_lags(cs, cfg.ar_lags)]

        for v in self.variables.monthly:
            if v not in g.columns:
                continue
            fallback = float(global_median.get(v, np.nanmedian(g[v])))
            xv = fill_series(g[v].to_numpy(np.float32), fallback, mode=self.fill_mode)
            blocks.append(monthly_windows(xv, L))

        for v in self.variables.quarterly:
            if v not in g.columns:
                continue
            fallback = float(global_median.get(v, np.nanmedian(g[v])))
            xv = fill_series(g[v].to_numpy(np.float32), fallback, mode=self.fill_mode)
            blocks.append(quarterly_taps(xv, pub_lag, Q))

        X_full = np.concatenate(blocks, axis=1).astype(np.float32) if blocks else np.empty((T, 0), dtype=np.float32)

        warm = max(
            max(cfg.ar_lags) if cfg.ar_lags else 0,
            (L - 1) if L > 0 else 0,
            (pub_lag + 3 * (Q - 1)) if Q > 0 else 0,
        )
        mask = np.isfinite(y_delta) & (np.arange(T) >= warm)

        # Group ids for selection bookkeeping:
        group_ids: list[np.int64] = []
        for _ in cfg.ar_lags:
            group_ids.append(np.int64(-1))
        for v in self.variables.monthly:
            gid = _stable_int64(f"M:{v}")
            group_ids += [gid] * L
        for v in self.variables.quarterly:
            gid = _stable_int64(f"Q:{v}")
            group_ids += [gid] * Q
        group_ids_arr = np.asarray(group_ids, dtype=np.int64)

        return DesignMatrix(
            X=X_full,
            y_delta=y_delta.astype(np.float32),
            dates=g["date"].to_numpy(),
            mask=mask,
            cs_level=cs,
            group_ids=group_ids_arr,
        )

    def pooled_design_up_to(
        self,
        *,
        train_end: pd.Timestamp,
        horizon: int,
        global_median: pd.Series,
    ):
        """Build pooled design up to a training end date.

        Returns
        -------
        X, y, groups, bonds, dates
        """
        cfg = self.config
        parts_X: list[np.ndarray] = []
        parts_y: list[np.ndarray] = []
        bond_ids: list[np.ndarray] = []
        row_dates: list[np.ndarray] = []

        groups: np.ndarray | None = None

        for cid, g in self.df.groupby("cusip", sort=False):
            g = g[g["date"] <= train_end].sort_values("date")
            if len(g) < cfg.min_train_min + 2:
                continue

            dm = self.build_design_for_bond(g, horizon=horizon, global_median=global_median)
            if not np.any(dm.mask):
                continue

            Xm = dm.X[dm.mask]
            ym = dm.y_delta[dm.mask]
            dts = dm.dates[dm.mask]

            parts_X.append(Xm)
            parts_y.append(ym)
            bond_ids.append(np.full(Xm.shape[0], str(cid), dtype=object))
            row_dates.append(dts)
            groups = dm.group_ids

        if not parts_X:
            return None

        X = np.vstack(parts_X).astype(np.float32)
        y = np.concatenate(parts_y).astype(np.float32)
        bonds = np.concatenate(bond_ids)
        dates = np.concatenate(row_dates)

        return X, y, groups, bonds, dates

    @staticmethod
    def month_end_series(obj) -> pd.Series:
        d = pd.to_datetime(obj, errors="coerce", dayfirst=True)
        s = pd.Series(d) if not isinstance(d, pd.Series) else d
        return s.dt.to_period("M").dt.to_timestamp("M")

    @staticmethod
    def month_end_scalar(ts) -> pd.Timestamp:
        return pd.Timestamp(pd.to_datetime(ts, errors="coerce", dayfirst=True)).to_period("M").to_timestamp("M")

    def t0_rows_for_bonds(
        self,
        *,
        bonds_req: Sequence[str],
        t0_scalar: pd.Timestamp,
        horizon: int,
        global_median: pd.Series,
    ):
        """Construct the design rows at time t0 for a list of bonds.

        This is used in the walk-forward loop, where predictions for date `t0 + H`
        are produced using information available at `t0`.
        """
        cfg = self.config
        t0_me = self.month_end_scalar(t0_scalar)
        H = int(horizon)
        warm_idx_req = max(
            max(cfg.ar_lags) if cfg.ar_lags else 0,
            (cfg.lm - 1) if cfg.lm > 0 else 0,
            (cfg.pub_lag + 3 * (cfg.q_taps - 1)) if cfg.q_taps > 0 else 0,
        )
        t_end = t0_me + pd.DateOffset(months=H)

        X_rows: list[np.ndarray] = []
        bond_list: list[str] = []

        for cid in bonds_req:
            g = self.df.loc[self.df["cusip"].astype(str) == str(cid)].sort_values("date")
            g = g[g["date"] <= t_end]
            if len(g) <= warm_idx_req:
                continue

            dm = self.build_design_for_bond(g, horizon=H, global_median=global_median)
            if dm.X.size == 0:
                continue

            d_me = self.month_end_series(dm.dates)
            hits = np.where(d_me.to_numpy() == np.datetime64(t0_me))[0]
            if hits.size == 0:
                continue
            pos = int(hits[0])
            if pos < warm_idx_req:
                continue

            X_rows.append(dm.X[pos : pos + 1, :])
            bond_list.append(str(cid))

        if not X_rows:
            return np.empty((0, 0), dtype=np.float64), np.array([], dtype=object)

        X_ref = np.vstack(X_rows).astype(np.float64)
        return X_ref, np.asarray(bond_list, dtype=object)
