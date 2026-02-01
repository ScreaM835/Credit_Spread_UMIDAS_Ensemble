Methodology
===========


.. note::
   For a publication-style, equation-heavy specification of the full pipeline
   (U‑MIDAS design, base learners, boosting, stacking, and regime‑aware GMM
   meta‑learning), see :doc:`paper`.

Overview
--------

The pipeline is a walk-forward (expanding window) forecasting framework for a panel of
bonds indexed by ``cusip``. The target is the **credit spread level** ``cs`` at monthly
frequency, and the model is trained to predict the *future change* in credit spreads
using a mixed-frequency U-MIDAS design.

Notation
--------

Let :math:`cs_{i,t}` denote the observed credit spread level for bond :math:`i` at
month-end :math:`t`. For a forecast horizon :math:`H` (in months), the supervised
learning target is the *change* (delta) in credit spreads:

.. math::

   y^{(H)}_{i,t} = cs_{i,t+H} - cs_{i,t}.

The level forecast is then reconstructed as:

.. math::

   \widehat{cs}_{i,t+H} = cs_{i,t} + \widehat{y}^{(H)}_{i,t}.

U-MIDAS feature construction
----------------------------

The feature vector for bond :math:`i` at time :math:`t` concatenates:

1. **Autoregressive lags of the level**
   (:math:`\{cs_{i,t-\ell}\}` for selected lags :math:`\ell`):

   - default lags: :math:`\ell \in \{1,2,3,6,12\}` months

2. **Monthly predictors** (for each monthly series :math:`x^{(m)}_{k,i,t}`):

   A fixed-length window of monthly lags is constructed with a one-month shift
   to avoid using contemporaneous values:

   .. math::

      \mathbf{x}^{(m)}_{k,i,t} =
      \big(x^{(m)}_{k,i,t-1}, x^{(m)}_{k,i,t-2}, \ldots, x^{(m)}_{k,i,t-L_M}\big),

   where :math:`L_M=6` by default.

3. **Quarterly predictors** (for each quarterly series :math:`x^{(q)}_{k,i,t}`):

   Quarterly variables are incorporated as *publication-lagged taps* spaced three
   months apart:

   .. math::

      \mathbf{x}^{(q)}_{k,i,t} =
      \big(x^{(q)}_{k,i,t-\lambda},
            x^{(q)}_{k,i,t-\lambda-3},
            \ldots,
            x^{(q)}_{k,i,t-\lambda-3(Q-1)}\big),

   where :math:`\lambda` is the publication lag (default :math:`\lambda=1`)
   and :math:`Q=4` by default.

The resulting design matrix is pooled across bonds and across all time points available
up to the training cut-off date.

Missing data handling
---------------------

The refactor preserves the notebook’s imputation behaviour by default:

- forward-fill within a series
- backward-fill within a series
- any remaining missing values are replaced by a median fallback

This is exposed via ``fill_mode``:

- ``ffill_bfill`` (default): matches the notebook
- ``ffill``: strictly causal (no backward fill)
- ``none``: no filling (fallback only)

For strictly causal experiments, ``ffill`` is recommended.

Walk-forward training loop
--------------------------

For each month-end :math:`t` in the evaluation period:

1. Train the model on all observations with dates :math:`\le t`
2. Predict :math:`\widehat{cs}_{i,t+1}` (or :math:`t+H` for general horizon :math:`H`)

The implementation follows an expanding-window scheme with a minimum start index
(``min_train_max``), mirroring the notebook.

Ensemble architecture
---------------------

Base learners
^^^^^^^^^^^^^

Four base learners are trained on the pooled U-MIDAS design:

- ElasticNet regression
- Ridge regression
- Gradient boosting (LightGBM if available; otherwise XGBoost)
- Shallow Random Forest

Feature standardisation
^^^^^^^^^^^^^^^^^^^^^^^

Features are scaled using ``RobustScaler``, which is less sensitive to outliers than
standard z-score scaling.

Meta-learning (stacking)
^^^^^^^^^^^^^^^^^^^^^^^^

A stacking meta-learner is fit on the base learners’ predictions using a holdout
validation split. The default meta model is a non-negative ridge regression, producing
a stable, interpretable non-negative combination of base learners.

An optional regime-aware mode is provided in which a Gaussian mixture model is fit on
residual-based regime features and separate meta-models are trained per regime.

Feature engineering option
--------------------------

The notebook introduces an optional ``enhanced`` feature mode that augments the U-MIDAS
design with additional heuristic features (e.g., limited interactions and short rolling
volatility proxies). This option is preserved for compatibility and can be disabled by
setting ``feature_engineering="basic"``.

Evaluation
----------

Two aggregation schemes are provided:

- **Micro metrics**: compute metrics over all prediction rows.
- **Macro metrics**: compute per-bond metrics, then average across bonds.

A **random-walk / no-change baseline** aligned with horizon :math:`H` is provided:

.. math::

   \widehat{cs}^{\text{RW}}_{i,t} = cs_{i,t-H}.

This corrects a common evaluation pitfall in which the baseline is accidentally merged
on the same month as the truth, resulting in a degenerate baseline error.

Outputs
-------

For each horizon :math:`H`, the pipeline writes a CSV file containing:

- ``cusip``
- ``date`` (the forecast target month-end)
- ``y_true`` (:math:`cs_{i,t}`)
- ``y_pred`` (:math:`\widehat{cs}_{i,t}`)
- ``cs_t0`` (:math:`cs_{i,t-H}`)
- ``y_delta_true`` (:math:`cs_{i,t} - cs_{i,t-H}`)
- ``y_delta_pred`` (:math:`\widehat{cs}_{i,t} - cs_{i,t-H}`)