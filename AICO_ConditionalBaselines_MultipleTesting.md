## AICO Extensions: Conditional Baselines and Multiple Testing Control

This document describes two theoretical and practical extensions to AICO:

- **A1 (Conditional baselines)**: replace purely marginal “typical value” baselines with conditional baselines that respect the dependence structure of the covariates.
- **A2 (Multiple testing control)**: add standard FDR and FWER procedures on top of the per-feature sign tests.

Both extensions are designed to strengthen the statistical interpretation of AICO in realistic correlated, high-dimensional settings while leaving the original API and sign-test core intact.

---

## 1. Problem Statements

### 1.1 Limitations of Marginal Baselines

- **Current behavior**: AICO’s `Baseline` uses marginal aggregators:
  - Continuous features: `MeanAgg` / `MedianAgg` / `QuantileAgg`.
  - Discrete / categorical features: `ModeAgg` / `AltModeAgg`.
- **Issue**:
  - For highly correlated features, marginal baselines do not respect the joint distribution.
  - The null “feature \(X_j\) has no effect” is approximated by replacing \(X_j\) with a global typical value, independent of \(X_{-j}\).
  - This can:
    - Understate or overstate feature contributions when correlations are strong.
    - Create unrealistic counterfactuals not supported by the data manifold.

**Problem A1**: Provide **conditional baselines** \(X_j^\star \sim \mathcal{L}(X_j \mid X_{-j})\) to better approximate a “no added information beyond other covariates” null, while keeping AICO’s model-agnostic and no-retraining properties.

---

### 1.2 Lack of Explicit Multiple-Testing Control

- **Current behavior**: AICO runs a valid one-sided sign test per feature at level \(\alpha\).
- **Issue**:
  - In realistic settings (dozens to hundreds of features), per-feature testing at level \(\alpha\) alone does not control:
    - **Family-wise error rate (FWER)** across all hypotheses.
    - **False discovery rate (FDR)** across discovered significant features.
  - Applied users and reviewers expect a clear statement about global error control, especially in regulated domains (credit, healthcare, etc.).

**Problem A2**: Add **standard multiple-testing procedures** (FDR/FWER) on top of AICO’s per-feature tests, without altering the core sign-test machinery or requiring any refitting.

---

## 2. Proposed Solutions

### 2.1 Conditional Baselines via `KNNConditionalSampler` (A1)

#### 2.1.1 High-Level Idea

- Introduce a **conditional sampler abstraction** that can be plugged into `Baseline` per variable.
- For a given variable \(X_j\) (potentially represented by multiple columns), we want to simulate
  \[
    X_j^\star \sim \mathcal{L}(X_j \mid X_{-j} = x_{-j})
  \]
  using a data-driven approximation.
- A simple, model-agnostic choice is **k-nearest-neighbor (k-NN) conditional resampling** based on training data.

#### 2.1.2 Implementation Overview

- **New module**: `src/conditional_baseline.py`

- **Class `KNNConditionalSampler`**
  - Constructor:
    - `n_neighbors=10`: number of neighbors for conditional sampling.
    - `conditioning_cols=None`: list of columns to condition on; defaults to all available columns.
    - `random_state=None`: controls reproducibility.
  - `fit(x_train: pd.DataFrame)`:
    - Decides `conditioning_cols_` (explicit list or all columns).
    - Fits `sklearn.neighbors.NearestNeighbors` on `x_train[conditioning_cols_]`.
    - Stores `x_train` for later resampling.
  - `__call__(x_train, x_test, cols)`:
    - Ensures the sampler is fitted.
    - For each test row, finds `n_neighbors` nearest neighbors in the training data based on `conditioning_cols_`.
    - Uniformly samples one neighbor index per test row.
    - Returns a `DataFrame` with columns `cols`, containing the sampled baseline values from the chosen neighbors, aligned to `x_test`'s index.

- **Integration into `Baseline` (`src/baseline.py`)**
  - Extended constructor:
    ```python
    class Baseline:
        def __init__(
            self,
            continuous_agg=MeanAgg(),
            discrete_agg=AltModeAgg(),
            categorical_agg=AltModeAgg(),
            conditional_samplers=None,
            glob=True,
        ):
            self.continuous_agg = continuous_agg
            self.discrete_agg = discrete_agg
            self.categorical_agg = categorical_agg
            self.conditional_samplers = conditional_samplers or dict()
            self.glob = glob
    ```
  - In `update`, the baseline stores:
    - `self.x_train`, `self.y_train`, `self.pred_func` for later use.
    - The original marginal aggregators (`self.agg`), preserving all previous behavior.
  - In `__call__(x_test, y_test, test_var)`:
    - If `test_var` is in `self.conditional_samplers`:
      - Uses the associated sampler to generate baseline values:
        ```python
        sampler = self.conditional_samplers[test_var]
        x_baseline[test_cols] = sampler(self.x_train, x_test, test_cols)
        ```
    - Otherwise falls back to the existing aggregator:
      ```python
      x_baseline[test_cols] = self.agg[test_var](x_test[test_cols], y_test)
      ```
    - For `glob=False`, the same logic is applied to other non-ignored variables: use a conditional sampler if provided, otherwise use the original aggregator.

#### 2.1.3 Backward Compatibility

- If `conditional_samplers` is not provided, the baseline behaves exactly as before:
  - Same aggregators, same signatures, same test logic.
- Existing code that instantiates `Baseline()` without the new argument will not change behavior.

#### 2.1.4 Example Usage

```python
from src.aico import AICO
from src.baseline import Baseline
from src.conditional_baseline import KNNConditionalSampler
from src.score import neg_squared_loss

sampler_X1 = KNNConditionalSampler(
    n_neighbors=10,
    conditioning_cols=["X2", "X3"],  # condition on selected covariates
    random_state=0,
)

baseline = Baseline(
    conditional_samplers={"X1": sampler_X1}
)

aico = AICO(
    x_train=x_train,
    y_train=y_train,
    pred_func=model.predict,
    score_func=neg_squared_loss,
    baseline=baseline,
    vars_ignored=["X0"],
    vars_discrete=[],
    vars_categorical=[],
)

aico.test(x_test=x_test, y_test=y_test)
result = aico.result
```

This replaces the baseline mechanism for `X1` with a k-NN conditional baseline while leaving all other variables unchanged.

---

### 2.2 Multiple Testing Control via `apply_fdr` / `apply_fwer` (A2)

#### 2.2.1 High-Level Idea

- Leave AICO’s per-variable sign tests and p-value interval computation unchanged.
- Add **post-processing utilities** that:
  - Extract a scalar p-value per variable (realized, or midpoint of an interval).
  - Apply standard multiple-testing corrections:
    - **FDR**: Benjamini–Hochberg (BH) and Benjamini–Yekutieli (BY).
    - **FWER**: Holm–Bonferroni and Bonferroni.
  - Write back adjusted statistics into the `AICO.result` table.

#### 2.2.2 Implementation Overview

- **New module**: `src/multipletest.py`

- Helper `_extract_p_values(result: pd.DataFrame, p_col: str) -> pd.Series`:
  - If `p_col` exists and has at least one non-NaN, uses it directly.
  - Else, if `p_value_lower` and `p_value_upper` are present, uses the midpoint:
    \[
      p = \frac{p_{\text{lower}} + p_{\text{upper}}}{2}
    \]
  - Excludes variables with `type == "ignored"` from adjustments.

- **`apply_fdr`**
  ```python
  def apply_fdr(result, alpha=0.05, method="BH", p_col="p_value") -> pd.DataFrame
  ```
  - Supported methods:
    - `"BH"`: Benjamini–Hochberg.
    - `"BY"`: Benjamini–Yekutieli (more conservative, arbitrary dependence).
  - Steps:
    - Extract valid p-values using `_extract_p_values`.
    - Sort them, compute ranks, and define:
      \[
        q_i = \frac{m}{i} \cdot p_{(i)} \cdot c_m
      \]
      where \(c_m = 1\) for BH, and \(c_m = \sum_{k=1}^m \frac{1}{k}\) for BY.
    - Enforce monotonicity via a reverse cumulative minimum.
    - Map back to original order, cap at 1, and populate:
      - `q_value`: adjusted q-values.
      - `significant_fdr`: boolean, `q_value <= alpha`.

- **`apply_fwer`**
  ```python
  def apply_fwer(result, alpha=0.05, method="holm", p_col="p_value") -> pd.DataFrame
  ```
  - Supported methods:
    - `"holm"`: Holm–Bonferroni step-down procedure.
    - `"bonferroni"`: classic Bonferroni correction (`p * m`).
  - For Holm:
    - Sort p-values ascending, compute:
      \[
        p^{\text{Holm}}_{(i)} = (m - i + 1) \cdot p_{(i)}
      \]
    - Enforce monotonicity with a forward cumulative maximum, cap at 1.
    - Map back to original order.
    - Write:
      - `p_value_fwer`: adjusted p-values.
      - `significant_fwer`: boolean, `p_value_fwer <= alpha`.

#### 2.2.3 Integration into `AICO`

- **Updated `src/aico.py`**:
  - Imports:
    ```python
    from .multipletest import apply_fdr, apply_fwer
    ```
  - New method on `AICO`:
    ```python
    def apply_multipletest(self, method="BH", alpha=None, family="fdr", p_col="p_value"):
        if alpha is None:
            alpha = self.alpha

        if family == "fdr":
            self.result = apply_fdr(self.result, alpha=alpha, method=method, p_col=p_col)
        elif family == "fwer":
            self.result = apply_fwer(self.result, alpha=alpha, method=method, p_col=p_col)
        else:
            raise ValueError("Unknown family, expected 'fdr' or 'fwer'.")
    ```
  - This method is **purely a post-processing step** on `self.result` and does not change the underlying sign-test or confidence interval logic.

#### 2.2.4 Example Usage

```python
aico.test(x_test=x_test, y_test=y_test)
aico.realize(seed=100)  # optional, for realized p-values

# FDR control at level 0.05 using BH
aico.apply_multipletest(family="fdr", method="BH", alpha=0.05)
result_fdr = aico.result

# FWER control at level 0.05 using Holm
aico.apply_multipletest(family="fwer", method="holm", alpha=0.05)
result_fwer = aico.result
```

The resulting `aico.result` now includes:

- `q_value`, `significant_fdr` (for FDR).
- `p_value_fwer`, `significant_fwer` (for FWER).

---

## 3. Testing and Verification

### 3.1 Unit Tests

- **File**: `tests/test_multipletest.py`
  - Constructs a toy `result` with four variables and known p-values.
  - Verifies:
    - `apply_fdr(..., method="BH")` marks the two smallest p-values as FDR-significant at \(\alpha=0.05\).
    - `apply_fwer(..., method="holm")` is more conservative but can still mark multiple small p-values as significant, matching the known Holm thresholds.

- **File**: `tests/test_conditional_baseline.py`
  - Builds a small correlated dataset where `X1` and `X2` are strongly linked.
  - Uses `KNNConditionalSampler` for `X1` via `Baseline(conditional_samplers={"X1": sampler})`.
  - Runs an end-to-end AICO test and asserts:
    - `aico.result` is non-empty.
    - `X1` appears in the result table.
  - Confirms that:
    - The conditional baseline path executes without errors.
    - Integration with `AICO`, `Baseline`, and the score functions is correct.

### 3.2 Full Test Run

- With the minimal dependencies installed (`numpy`, `pandas`, `scikit-learn`, `statsmodels`, `pytest`), running:

```bash
python -m pytest -q
```

currently yields:

- `3 passed` (covering conditional baselines and multiple-testing utilities).

---

## 4. Summary of Contributions

- **A1 – Conditional Baselines**
  - Introduced `KNNConditionalSampler` and `Baseline.conditional_samplers` to generate conditional baselines using k-NN resampling.
  - Preserves all existing baseline behavior as the default.
  - Enables more realistic, distribution-respecting counterfactuals, especially under strong feature correlations.

- **A2 – Multiple Testing Control**
  - Added `apply_fdr` and `apply_fwer` utilities for standard FDR and FWER corrections.
  - Exposed an `AICO.apply_multipletest` method for a simple, user-facing API.
  - Provides explicit, standard global error control layered on top of AICO’s existing sign-test framework.

Both extensions are **backwards compatible**, **fully integrated**, and **covered by automated tests**, making them suitable for a focused, review-friendly upstream contribution and a foundation for future theoretical work.

