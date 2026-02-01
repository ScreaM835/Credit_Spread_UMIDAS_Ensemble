"""Advanced Ensemble U-MIDAS.

This package provides a reproducible implementation of the walk-forward
ensemble U-MIDAS pipeline contained in the original research notebook.

Key components:
- Mixed-frequency U-MIDAS feature construction (monthly windows, quarterly taps, AR lags)
- Walk-forward (expanding window) training and prediction
- Base-learner ensemble with stacking meta-learning
- Evaluation metrics and diagnostic visualisations
"""

from __future__ import annotations

__all__ = [
    "__version__",
]

__version__ = "0.1.0"
