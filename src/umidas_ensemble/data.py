from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from .logging_utils import get_logger

logger = get_logger(__name__)


def load_panel_data(data_path: Path) -> pd.DataFrame:
    """Load the panel dataset.

    The dataset is expected to contain at least:
    - `cusip`: bond identifier
    - `date`: timestamp
    - `cs`: target series (credit spread level)

    The loader mirrors the notebook behaviour:
    - parse `date`
    - drop missing {cusip,date,cs}
    - drop duplicate (cusip,date) pairs
    - sort by (cusip,date)
    - cast cusip to string
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    df = pd.read_csv(data_path, low_memory=False)
    if "date" not in df.columns:
        raise ValueError("Input data must contain a 'date' column")
    if "cusip" not in df.columns:
        raise ValueError("Input data must contain a 'cusip' column")
    if "cs" not in df.columns:
        raise ValueError("Input data must contain a 'cs' column")

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)
    df = (
        df.dropna(subset=["cusip", "date", "cs"])
        .drop_duplicates(subset=["cusip", "date"], keep="first")
        .sort_values(["cusip", "date"])
        .reset_index(drop=True)
    )
    df["cusip"] = df["cusip"].astype(str)
    return df


def _normalize_freq(x) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s in {"m", "mon", "month", "monthly", "mth"}:
        return "Monthly"
    if s in {"q", "qr", "qtr", "quarter", "quarterly"}:
        return "Quarterly"
    if "month" in s:
        return "Monthly"
    if "quarter" in s:
        return "Quarterly"
    return None


def _freq_from_sheetname(sheet: str) -> str | None:
    s = str(sheet).lower()
    if "monthly" in s or re.search(r"\bmon(th(ly)?)?\b", s):
        return "Monthly"
    if "quarterly" in s or re.search(r"\bq(uart(er(ly)?)?)?\b", s):
        return "Quarterly"
    return None


def build_variable_lists_from_excel(
    *,
    df: pd.DataFrame,
    dict_path: Path,
    cache_path: Path | None = None,
) -> Tuple[list[str], list[str]]:
    """Infer monthly and quarterly variable lists from an Excel dictionary.

    The notebook uses a heuristic procedure:
    - for each sheet, find the column that contains the greatest number of symbols
      that match columns in `df` (on a small header sample);
    - infer frequency from (i) the most frequency-like column in the sheet, or
      (ii) the sheet name (fallback);
    - keep only variables that are numeric in `df`;
    - exclude the target `cs` from monthly variables.

    Parameters
    ----------
    df:
        Panel DataFrame loaded with :func:`load_panel_data`.
    dict_path:
        Path to the Excel variable dictionary.
    cache_path:
        If provided, write a JSON cache with the inferred lists.

    Returns
    -------
    monthly_vars_all, quarterly_vars_all
        Lists of column names present in `df`.
    """
    dict_path = Path(dict_path)
    if not dict_path.exists():
        raise FileNotFoundError(f"Missing variable dictionary: {dict_path}")

    # Use only a small slice to identify candidate columns, mirroring the notebook.
    panel_head = df.head(500)
    panel_cols = set(panel_head.columns)

    xls = pd.ExcelFile(dict_path, engine="openpyxl")
    rows: list[dict] = []

    for sheet in xls.sheet_names:
        tab = xls.parse(sheet)
        if tab.shape[1] == 0:
            continue

        # Choose the column whose values best match the panel column names.
        best_sym = None
        best_hits = -1
        for c in tab.columns:
            vals = tab[c].dropna().astype(str).str.strip()
            hits = sum(v in panel_cols for v in vals)
            if hits > best_hits:
                best_hits = hits
                best_sym = c

        if best_sym is None or best_hits <= 0:
            continue

        # Choose the column that looks most like a frequency indicator.
        freq_col = None
        freq_share = 0.0
        for c in tab.columns:
            vals = tab[c].dropna()
            tot = len(vals)
            ok = sum(_normalize_freq(v) in {"Monthly", "Quarterly"} for v in vals) if tot > 0 else 0
            share = ok / tot if tot > 0 else 0.0
            if share > freq_share:
                freq_share = share
                freq_col = c

        sheet_hint = _freq_from_sheetname(sheet)

        syms = tab[best_sym].dropna().astype(str).str.strip()
        for i, sym in syms.items():
            if sym not in panel_cols:
                continue

            fr = _normalize_freq(tab.loc[i, freq_col]) if (freq_col is not None) else None
            if fr is None:
                fr = sheet_hint
            if fr not in {"Monthly", "Quarterly"}:
                continue
            rows.append({"symbol": sym, "frequency": fr})

    if not rows:
        raise ValueError(
            "Could not extract (symbol, frequency) pairs from the Excel dictionary. "
            "Check the sheet structure and column naming."
        )

    mapping = pd.DataFrame(rows).drop_duplicates(subset=["symbol"], keep="last")

    # Keep only numeric series present in df.
    def _is_numeric(col: str) -> bool:
        if col not in df.columns:
            return False
        s = df[col].dropna().infer_objects()
        if s.empty:
            return False
        return np.issubdtype(s.dtype, np.number)

    mapping["is_numeric"] = mapping["symbol"].apply(_is_numeric)
    usable = mapping.query("is_numeric").copy()

    monthly_vars_all = sorted(usable.query('frequency=="Monthly" and symbol!="cs"')["symbol"].unique().tolist())
    quarterly_vars_all = sorted(usable.query('frequency=="Quarterly"')["symbol"].unique().tolist())

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "monthly_vars_all": monthly_vars_all,
            "quarterly_vars_all": quarterly_vars_all,
            "counts": {"monthly": len(monthly_vars_all), "quarterly": len(quarterly_vars_all)},
        }
        cache_path.write_text(json.dumps(payload, indent=2))
        logger.info("Wrote variable list cache to %s", cache_path)

    return monthly_vars_all, quarterly_vars_all


def load_variable_lists(cache_path: Path) -> Tuple[list[str], list[str]]:
    """Load variable lists from a JSON cache."""
    cache_path = Path(cache_path)
    payload = json.loads(cache_path.read_text())
    return payload["monthly_vars_all"], payload["quarterly_vars_all"]
