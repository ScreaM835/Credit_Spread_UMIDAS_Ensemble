Documentation
==================================

Title
-----

**Advanced Ensemble U‑MIDAS for Panel Credit‑Spread Forecasting**

Abstract
--------

This package implements an expanding‑window, walk‑forward forecasting pipeline for a
large panel of bond credit spreads indexed by ``cusip``. The predictive target is the
**future change** in credit spread levels over a horizon of :math:`H` months.
Predictors are mixed‑frequency (monthly and quarterly) and are mapped into a common
monthly grid via a U‑MIDAS (unrestricted MIDAS) design. The predictive model is a
stacked ensemble combining regularised linear models, a shallow random forest, and
gradient‑boosted trees. The stacking layer supports an optional **regime‑aware**
mode that detects regimes using a Gaussian mixture model (GMM) on residual and
volatility features and fits regime‑specific stacking weights.

This document provides a publication‑style description of the model specification,
training protocol, and evaluation methodology implemented in the package.

1. Data and preprocessing
-------------------------

1.1 Panel structure and calendar alignment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The raw dataset is a panel:

- bond identifier :math:`i \in \{1,\dots,N\}` (``cusip``),
- calendar date :math:`d` (irregular, but treated as monthly),
- credit spread level :math:`cs_{i}(d)` and a set of predictors.

All observations are mapped to **month‑end timestamps**. Define the month‑end operator
:math:`\mathrm{ME}(\cdot)` as the last calendar day of the month containing :math:`d`:

.. math::
   t = \mathrm{ME}(d)

The pipeline works on the month‑end grid :math:`t \in \mathcal{T}` after:

- dropping rows with missing ``cusip``/``date``/``cs``;
- dropping duplicate keys :math:`(i,t)`;
- sorting within each bond by increasing :math:`t`.

1.2 Mixed‑frequency predictor sets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A variable dictionary is used to partition predictors into two sets:

- Monthly predictors: :math:`\mathcal{X}^{(m)} = \{x^{(m)}_{k}\}_{k=1}^{K_m}`
- Quarterly predictors: :math:`\mathcal{X}^{(q)} = \{x^{(q)}_{k}\}_{k=1}^{K_q}`

Only numeric predictors that appear in the panel are retained.

2. Forecasting task
-------------------

2.1 Horizon‑:math:`H` delta target
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`cs_{i,t}` be the credit spread level for bond :math:`i` at month‑end :math:`t`.
For a horizon :math:`H \in \mathbb{N}` months, the supervised learning target is the
**change** (delta) in spreads:

.. math::

   y^{(H)}_{i,t} = cs_{i,t+H} - cs_{i,t}.

The model produces :math:`\widehat{y}^{(H)}_{i,t}` and reconstructs the level forecast:

.. math::

   \widehat{cs}_{i,t+H} = cs_{i,t} + \widehat{y}^{(H)}_{i,t}.

2.2 Walk‑forward prediction date indexing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The saved prediction files index predictions by the **target date** :math:`t` and store:

- ``y_true`` := :math:`cs_{i,t}` (true level at target month‑end),
- ``y_pred`` := :math:`\widehat{cs}_{i,t}` (predicted level),

where the design matrix is constructed at the origin month :math:`t_0 = t - H`.

3. U‑MIDAS feature construction (mixed frequency)
-------------------------------------------------

For each bond :math:`i` and origin month :math:`t`, the U‑MIDAS feature vector
:math:`\mathbf{x}_{i,t} \in \mathbb{R}^p` concatenates (i) autoregressive lags of
the level, (ii) lag blocks for monthly predictors, and (iii) publication‑lagged taps
for quarterly predictors.

3.1 Autoregressive lags of the level
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a set of AR lags :math:`\mathcal{L} = \{\ell_1,\dots,\ell_{K_{ar}}\}` (months), define:

.. math::

   \mathbf{x}^{(ar)}_{i,t} =
   \big(cs_{i,t-\ell_1}, cs_{i,t-\ell_2}, \ldots, cs_{i,t-\ell_{K_{ar}}}\big).

The default notebook configuration uses:

.. math::

   \mathcal{L} = \{1,2,3,6,12\}.

3.2 Monthly predictor lag windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each monthly predictor :math:`x^{(m)}_{k,i,t}`, define a fixed window of length
:math:`L_M` with a one‑month shift to avoid contemporaneous use:

.. math::

   \mathbf{x}^{(m)}_{k,i,t} =
   \big(x^{(m)}_{k,i,t-1}, x^{(m)}_{k,i,t-2}, \ldots, x^{(m)}_{k,i,t-L_M}\big).

Default: :math:`L_M = 6`.

3.3 Quarterly predictor publication‑lagged taps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Quarterly predictors are incorporated via :math:`Q` taps spaced at 3‑month intervals,
with an explicit publication lag :math:`\lambda` months:

.. math::

   \mathbf{x}^{(q)}_{k,i,t} =
   \big(x^{(q)}_{k,i,t-\lambda},
        x^{(q)}_{k,i,t-\lambda-3},
        \ldots,
        x^{(q)}_{k,i,t-\lambda-3(Q-1)}\big).

Defaults: :math:`\lambda = 1`, :math:`Q = 4`.

3.4 Full feature vector and dimensionality
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The final U‑MIDAS feature vector is:

.. math::

   \mathbf{x}_{i,t} =
   \Big[
      \mathbf{x}^{(ar)}_{i,t},
      \{\mathbf{x}^{(m)}_{k,i,t}\}_{k=1}^{K_m},
      \{\mathbf{x}^{(q)}_{k,i,t}\}_{k=1}^{K_q}
   \Big].

Its dimension is:

.. math::

   p = K_{ar} + K_m \cdot L_M + K_q \cdot Q.

4. Missing data handling
------------------------

Within each bond series and each predictor series, the notebook’s default imputation
is:

1. forward fill (FFILL),
2. backward fill (BFILL),
3. replace any remaining missing values with a median fallback.

Denote an observed series :math:`z_t` with missing values. Define the imputed series
:math:`\tilde{z}_t` as:

.. math::

   \tilde{z}_t =
   \begin{cases}
     \mathrm{FFILL\_BFILL}(z)_t, & \text{if defined} \\
     \mathrm{median}(z), & \text{otherwise.}
   \end{cases}

**Important:** the BFILL step is not strictly causal and can introduce look‑ahead
information when there are internal gaps. The package therefore exposes a strictly
causal option (FFILL only).

5. Feature scaling
------------------

Let :math:`X \in \mathbb{R}^{n \times p}` be the pooled design matrix.
Robust scaling is applied feature‑wise using the median and inter‑quartile range (IQR):

.. math::

   X'_{:,j} = \frac{X_{:,j} - \mathrm{median}(X_{:,j})}{\mathrm{IQR}(X_{:,j}) + \varepsilon},

with a small :math:`\varepsilon>0` for numerical stability. This corresponds to
``sklearn.preprocessing.RobustScaler``.

6. Base learners
----------------

The ensemble contains four base learners trained on the pooled U‑MIDAS design.

6.1 Ridge regression
^^^^^^^^^^^^^^^^^^^^

Ridge estimates coefficients :math:`\beta \in \mathbb{R}^p` by minimising:

.. math::

   \min_{\beta}
   \frac{1}{n}\sum_{r=1}^{n} \big(y_r - \mathbf{x}_r^\top \beta\big)^2
   + \alpha \lVert \beta \rVert_2^2.

6.2 Elastic Net regression
^^^^^^^^^^^^^^^^^^^^^^^^^^

Elastic Net combines :math:`\ell_1` and :math:`\ell_2` penalties:

.. math::

   \min_{\beta}
   \frac{1}{n}\sum_{r=1}^{n} \big(y_r - \mathbf{x}_r^\top \beta\big)^2
   + \alpha\left(
     \rho \lVert \beta \rVert_1 + \frac{1-\rho}{2}\lVert \beta \rVert_2^2
   \right),

where :math:`\alpha>0` controls overall regularisation and :math:`\rho \in [0,1]`
controls sparsity.

6.3 Random Forest regression (bagging)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A random forest is an average of :math:`B` regression trees:

.. math::

   \widehat{f}_{RF}(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^{B} T_b(\mathbf{x}),

where each :math:`T_b` is trained on a bootstrap sample and uses random feature
subsampling at splits (``max_features``).

6.4 Gradient‑boosted decision trees (GBDT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The boosting model is an additive expansion of trees:

.. math::

   F_M(\mathbf{x}) = \sum_{m=0}^{M} \nu \, \gamma_m \, h_m(\mathbf{x}),

where:

- :math:`h_m(\cdot)` is a regression tree at iteration :math:`m`,
- :math:`\nu \in (0,1]` is the learning rate,
- :math:`\gamma_m` is a step size.

The notebook config uses an absolute‑error loss (L1), via LightGBM
``objective="regression_l1"`` or XGBoost ``objective="reg:absoluteerror"``.
In gradient boosting, the next tree is fitted to the negative gradient of the loss.
For L1 loss :math:`\ell(y,F)=|y-F|`, the (sub)gradient with respect to :math:`F` is:

.. math::

   \frac{\partial \ell}{\partial F} = -\mathrm{sign}(y-F),

so the pseudo‑residuals are approximately :math:`r = \mathrm{sign}(y-F)`.


A more explicit boosting iteration can be written as:

.. math::

   \begin{aligned}
   F_0(\mathbf{x}) &= \arg\min_{c\in\mathbb{R}} \sum_{r=1}^{n} \ell(y_r,c), \\
   r_r^{(m)} &= -\left.\frac{\partial \ell(y_r,F(\mathbf{x}_r))}{\partial F}\right|_{F=F_{m-1}}, \\
   h_m &\approx \arg\min_{h\in\mathcal{H}} \sum_{r=1}^{n} \big(r_r^{(m)} - h(\mathbf{x}_r)\big)^2, \\
   F_m(\mathbf{x}) &= F_{m-1}(\mathbf{x}) + \nu \, \gamma_m \, h_m(\mathbf{x}),
   \end{aligned}

where :math:`\mathcal{H}` is the function class of regression trees and :math:`\gamma_m`
is a line‑search or leaf‑wise step size (implementation‑specific).
For L1 loss, :math:`F_0` is the sample median and :math:`r^{(m)}` is a sign‑based
(sub)gradient.


7. Stacking meta‑learning
-------------------------

Let the base learners be indexed by :math:`b \in \{1,\dots,B\}`.
For each training row :math:`r`, define base predictions:

.. math::

   \widehat{y}^{(b)}_r = f_b(\mathbf{x}_r).

Stacking constructs the meta‑feature vector:

.. math::

   \mathbf{z}_r = \big(\widehat{y}^{(1)}_r, \ldots, \widehat{y}^{(B)}_r\big).

7.1 Holdout stacking protocol
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The implementation uses a holdout validation split of the pooled training set:

- Fit base learners on the training subset.
- Compute base predictions :math:`\mathbf{z}_r` on the validation subset.
- Fit a meta‑model mapping :math:`\mathbf{z}_r \mapsto y_r`.

7.2 Non‑negative ridge stacking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default meta‑model is a ridge regression with non‑negative weights:

.. math::

   \min_{\mathbf{w} \ge 0}
   \frac{1}{n_v}\sum_{r\in\mathcal{V}} \big(y_r - \mathbf{z}_r^\top \mathbf{w}\big)^2
   + \lambda \lVert \mathbf{w}\rVert_2^2.

The positivity constraint yields a stable, interpretable convex combination of base
models.

8. Regime‑aware meta‑learning (GMM clustering)
----------------------------------------------

The pipeline supports a regime‑aware stacking mode in which regimes are inferred from
meta‑level residual dynamics.

8.1 Regime features
^^^^^^^^^^^^^^^^^^^

Let :math:`\bar{y}_r = \frac{1}{B}\sum_{b=1}^{B}\widehat{y}^{(b)}_r` be the mean base
prediction. Define a residual proxy:

.. math::

   e_r = y_r - \bar{y}_r.

Define a rolling volatility proxy (as implemented) over the residual series:

.. math::

   \sigma_r = \mathrm{Std}\big(e_{r-11}, \ldots, e_r\big),

with the understanding that the rolling window is applied over the pooled row order
as produced by the training set construction.

The regime feature vector is:

.. math::

   \mathbf{s}_r = \big(e_r,\sigma_r\big) \in \mathbb{R}^2.

8.2 Gaussian mixture model
^^^^^^^^^^^^^^^^^^^^^^^^^^

A :math:`K`‑component Gaussian mixture model is fitted to :math:`\{\mathbf{s}_r\}`:

.. math::

   p(\mathbf{s})
   = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{s}\mid \mu_k,\Sigma_k),

where :math:`\pi_k \ge 0`, :math:`\sum_k \pi_k = 1`.
Parameters are estimated by maximum likelihood (typically via EM).
Each row is assigned a regime label:


In an EM formulation, the E‑step computes posterior responsibilities:

.. math::

   \tau_{r,k}
   = p(z_r=k\mid \mathbf{s}_r)
   = \frac{\pi_k \,\mathcal{N}(\mathbf{s}_r\mid \mu_k,\Sigma_k)}
          {\sum_{j=1}^{K} \pi_j \,\mathcal{N}(\mathbf{s}_r\mid \mu_j,\Sigma_j)}.

The M‑step updates the mixture parameters:

.. math::

   \begin{aligned}
   N_k &= \sum_{r=1}^{n} \tau_{r,k}, \\
   \pi_k &= \frac{N_k}{n}, \\
   \mu_k &= \frac{1}{N_k}\sum_{r=1}^{n} \tau_{r,k}\mathbf{s}_r, \\
   \Sigma_k &= \frac{1}{N_k}\sum_{r=1}^{n} \tau_{r,k}(\mathbf{s}_r-\mu_k)(\mathbf{s}_r-\mu_k)^\top.
   \end{aligned}


.. math::

   \widehat{z}_r = \arg\max_{k} \, p(k \mid \mathbf{s}_r).

Default: :math:`K=3`.

8.3 Regime‑specific stacking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each regime :math:`k`, a separate meta‑model is fitted on the subset
:math:`\mathcal{V}_k = \{r\in\mathcal{V}:\widehat{z}_r=k\}`:

.. math::

   \widehat{\mathbf{w}}_k
   = \arg\min_{\mathbf{w} \ge 0}
     \sum_{r\in\mathcal{V}_k} \big(y_r - \mathbf{z}_r^\top \mathbf{w}\big)^2
     + \lambda \lVert \mathbf{w}\rVert_2^2.

At prediction time, if a current regime feature vector :math:`\mathbf{s}_*` is
available, the GMM assigns :math:`\widehat{z}_*` and the corresponding meta‑model is
used. If no regime is provided (or assignment fails), the implementation falls back
to averaging regime‑specific predictions.

9. Optional engineered features
-------------------------------

An optional ``feature_mode="enhanced"`` adds heuristic transformations to the U‑MIDAS
matrix.

Let :math:`X \in \mathbb{R}^{n\times p}` be the pooled U‑MIDAS matrix.

9.1 AR “momentum” across lag columns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the first :math:`p_{ar}` columns (default :math:`p_{ar}=\min(5,p)`), define the
difference across adjacent lag columns (with a prepend convention):

.. math::

   M_{r,1} = 0,\qquad
   M_{r,j} = X_{r,j} - X_{r,j-1},\quad j=2,\ldots,p_{ar}.

9.2 Row‑local rolling standard deviation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A short rolling standard deviation is computed across rows:

.. math::

   S_{r,:} = \mathrm{Std}\big(X_{r-3,:},\ldots,X_{r,:}\big),

with :math:`S_{r,:}=0` for early rows. The implementation retains the five columns
with the highest average :math:`S_{:,j}`.

9.3 Limited pairwise interactions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`\mathcal{J}` be the set of the most variant columns. Interactions are:

.. math::

   I^{(j,k)}_{r} = X_{r,j}\,X_{r,k},\qquad (j,k)\in\mathcal{J}\times\mathcal{J},\ j<k,

subject to a cap (10 interactions) for computational tractability.

10. Expanding‑window training protocol
--------------------------------------

For each target month :math:`t` in the evaluation period, define the origin month
:math:`t_0=t-H`. Training data consist of all bond‑month rows with origin dates
:math:`\tau \le t_0` for which both feature lags and the horizon target
:math:`y^{(H)}_{i,\tau}` exist.

Let :math:`\mathcal{D}_{t_0}` denote the pooled training set:

.. math::

   \mathcal{D}_{t_0} = \{(\mathbf{x}_{i,\tau}, y^{(H)}_{i,\tau}) : \tau \le t_0\}.

At each step:

1. construct :math:`\mathcal{D}_{t_0}`,
2. robust‑scale features,
3. fit base learners on a training subset,
4. fit the meta learner on a holdout subset,
5. produce :math:`\widehat{y}^{(H)}_{i,t_0}` and reconstruct :math:`\widehat{cs}_{i,t}`.


Algorithm 1: Expanding‑window stacked U‑MIDAS ensemble
------------------------------------------------------

The following describes the walk‑forward training and prediction loop at a high level.

.. code-block:: text

   Inputs:
     Raw panel data {cs_{i,t}, X_{i,t}} for i=1..N, t in month-ends
     Horizon H
     Feature hyperparameters (AR lags, L_M, lambda, Q)
     Base models {f_b}_{b=1..B}
     Meta method (stacking_ridge or regime_aware)

   For each target month-end t in evaluation range:
     t0 = t - H                                  # origin month
     Build pooled training set D_{t0}:
       for each bond i:
         for each origin tau <= t0 with tau+H observed:
           compute x_{i,tau} via U-MIDAS
           compute y_{i,tau}^{(H)} = cs_{i,tau+H} - cs_{i,tau}
     Robust-scale pooled features
     Split pooled data into train/validation (holdout)
     Fit each base model f_b on training subset
     Compute base predictions on validation subset -> meta features z_r
     Fit meta model on (z_r, y_r):
       - stacking_ridge: non-negative ridge on z_r
       - regime_aware: fit GMM on (residual, volatility) and fit regime-specific meta models
     For each bond i observable at t:
       compute x_{i,t0}
       predict delta \hat{y}_{i,t0}^{(H)} via stacked ensemble
       output level forecast \hat{cs}_{i,t} = cs_{i,t0} + \hat{y}_{i,t0}^{(H)}

   Output: prediction table (cusip, date=t, y_true=cs_{i,t}, y_pred=\hat{cs}_{i,t})


11. Evaluation metrics
----------------------

Let the prediction dataset contain rows indexed by :math:`r=1,\dots,n` with values
:math:`y_r` (true level) and :math:`\hat{y}_r` (predicted level). Define residuals
:math:`e_r=y_r-\hat{y}_r`.

11.1 Micro‑averaged metrics (pooled)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Micro mean squared error (MSE) and root MSE:

.. math::

   \mathrm{MSE}_{\mu} = \frac{1}{n}\sum_{r=1}^{n} e_r^2,\qquad
   \mathrm{RMSE}_{\mu} = \sqrt{\mathrm{MSE}_{\mu}}.

Micro mean absolute error (MAE):

.. math::

   \mathrm{MAE}_{\mu} = \frac{1}{n}\sum_{r=1}^{n} |e_r|.

Micro :math:`R^2`:

.. math::

   R^2_{\mu} = 1 - \frac{\sum_r e_r^2}{\sum_r (y_r - \bar{y})^2},\qquad
   \bar{y}=\frac{1}{n}\sum_r y_r.

Percentage absolute error (PAE) used in the notebook divides by :math:`|y_r|`:

.. math::

   \mathrm{PAE}_{\mu} = \frac{100}{n}\sum_{r=1}^{n} \frac{|e_r|}{|y_r|+\varepsilon}.

Signed percentage error (MPE) uses the same denominator:

.. math::

   \mathrm{MPE}_{\mu} = \frac{100}{n}\sum_{r=1}^{n} \frac{e_r}{|y_r|+\varepsilon}.

Symmetric MAPE (sMAPE):

.. math::

   \mathrm{sMAPE}_{\mu}
   = \frac{100}{n}\sum_{r=1}^{n} \frac{2|e_r|}{|y_r|+|\hat{y}_r|+\varepsilon}.

Relative RMSE (rRMSE) normalises by :math:`\mathbb{E}|y|`:

.. math::

   \mathrm{rRMSE}_{\mu} = 100\,\frac{\mathrm{RMSE}_{\mu}}{\frac{1}{n}\sum_r |y_r|+\varepsilon}.

11.2 Macro‑averaged metrics (across bonds)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Macro aggregation computes metrics per bond and averages them across bonds.

Let :math:`\mathcal{R}_i` be the set of rows for bond :math:`i`. For any metric
functional :math:`\phi(\cdot)`, define:

.. math::

   \phi_{\mathrm{macro}} = \frac{1}{N}\sum_{i=1}^{N} \phi\big(\{(y_r,\hat{y}_r):r\in\mathcal{R}_i\}\big).

**Note on macro** :math:`R^2`: if a bond has near‑constant :math:`y_r` over its test
rows, :math:`\sum_{r\in\mathcal{R}_i}(y_r-\bar{y}_i)^2` can be extremely small and
:math:`R^2_i` can become numerically unstable, resulting in very large negative values
that dominate the mean.

11.3 Baseline (random walk)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A meaningful no‑change baseline for level forecasting at horizon :math:`H` is:

.. math::

   \widehat{cs}^{RW}_{i,t} = cs_{i,t-H}.

This is the baseline implemented in the package’s evaluation utilities.

12. Reproducibility notes
-------------------------

- The package preserves the notebook’s feature construction, model family choices,
  and stacking logic.
- Dataset paths are provided via CLI arguments or configuration objects.
- For strictly causal studies, disable backward filling (``fill_mode="ffill"``) and
  compute medians only within the training window.

References
----------

- Breiman, L. (2001). Random Forests.
- Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine.
- Ghysels, E., Santa‑Clara, P., and Valkanov, R. (2004). The MIDAS touch: Mixed Data Sampling regression models.

