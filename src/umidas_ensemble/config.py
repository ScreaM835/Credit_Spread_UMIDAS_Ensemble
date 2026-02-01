from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class UMIDASConfig:
    """Configuration for U-MIDAS feature construction.

    Parameters follow the original notebook as defaults.

    Notes
    -----
    - `lm` controls the number of monthly lags ("taps") per monthly variable.
    - `q_taps` controls the number of quarterly taps, spaced 3 months apart.
    - `pub_lag` implements a publication lag (in months) for quarterly variables.
    - `ar_lags` are lags of the dependent variable level `cs` included as predictors.
    """

    lm: int = 6
    q_taps: int = 4
    pub_lag: int = 1
    ar_lags: tuple[int, ...] = (1, 2, 3, 6, 12)

    winsor_q: float = 0.01

    # Minimum observations per bond in the pooled training design.
    min_train_min: int = 24

    # Index of the first training end-date in the walk-forward loop
    # (expressed as an index into the sorted set of month-end dates).
    min_train_max: int = 80


@dataclass(frozen=True)
class EnsembleConfig:
    """Configuration for the ensemble training/prediction loop."""

    horizons: tuple[int, ...] = (1,)

    meta_learning_method: str = "stacking_ridge"  # stacking_ridge | stacking_elastic | neural_meta | regime_aware
    feature_engineering: str = "enhanced"  # basic | enhanced

    validation_months: int = 6
    random_state: int = 42

    # Operational settings
    resume: bool = True
    log_every: int = 1
    n_threads: int = 2
    n_jobs: int = 4


@dataclass(frozen=True)
class PathsConfig:
    """I/O paths for running the pipeline."""

    data_path: Path
    dict_path: Path | None = None  # Optional variable dictionary (Excel)
    save_dir: Path = Path("results")
    variable_list_cache: Path | None = None

    def __post_init__(self) -> None:
        # Normalise to Path
        object.__setattr__(self, "data_path", Path(self.data_path))
        if self.dict_path is not None:
            object.__setattr__(self, "dict_path", Path(self.dict_path))
        object.__setattr__(self, "save_dir", Path(self.save_dir))
        if self.variable_list_cache is not None:
            object.__setattr__(self, "variable_list_cache", Path(self.variable_list_cache))


@dataclass(frozen=True)
class VariableLists:
    """Monthly and quarterly variable lists."""

    monthly: tuple[str, ...]
    quarterly: tuple[str, ...]

    @classmethod
    def from_iterables(cls, monthly: Iterable[str], quarterly: Iterable[str]) -> "VariableLists":
        return cls(monthly=tuple(monthly), quarterly=tuple(quarterly))
