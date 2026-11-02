# Python / Patterns in the Record

This is the Python reference implementation. The R edition reproduces every
deterministic table, and `scripts/check_parity.py` verifies that they match.
Read the root `METHODOLOGY.md` before interpreting any result.

The input is 1,309 titanic3 passengers; crew are not included. The audit also
reads a clearly labelled **synthetic** promotion dataset.

## Run on Windows (Python 3.11)

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m Python.run_analysis        # about 90 s
.\.venv\Scripts\python.exe -m pytest tests\test_python_analysis.py tests\test_audit.py
.\.venv\Scripts\python.exe -m Python.make_notebook
.\.venv\Scripts\python.exe -m Python.execute_notebook
```

- After the dependencies are installed, no runtime internet access is needed.
- Forest parallelism is bounded at two workers.
- `outputs/python/requirements-resolved.txt` records the exact environment.

## Modules

### `analysis.py`: the original lenses

This module uses the 1,048 / 261 holdout.

- **Rates:** Beta(1,1) sex × class rates.
- **Rules:** Apriori on training rows only.
- **Models:** a prevalence baseline, a 500-tree forest and a Laplace Bayesian
  logistic model.
- **Diagnostics:** right-closed five-bin calibration.
- **Fare:** per-person throughout.

### `audit.py`: the people-science audit

- **Feature-set comparisons:** policy, structure, combined, sex-blind and
  sex-blind + title. All are Laplace logistic models, with a random-forest
  benchmark. "Policy" names a demographic feature set, not a recovered policy.
- **Evaluation:** 20 × 5 repeated grouped cross-validation with Nadeau–Bengio
  paired intervals.
- **Proxy tests:** the proxy gap and sex recovery.
- **Group diagnostics:** rate ratios, agreement in group averages and error
  rates, with small cells suppressed. Average agreement is not full calibration.
- **Descriptive history:** the policy groups, child cutoffs, boat side and
  missing age.
- **Synthetic simulation:** checks the audit against the known injected bias.

## Outputs (`outputs/python/`)

All CSVs are written at full precision; empty cells are NaN, meaning
suppressed or undefined.

| Group | Files |
|---|---|
| Original lenses | `metrics`, `predictions`, `cv_predictions`, `calibration`, `coefficients`, `rules`, `bayesian_rates`, `missingness`, `design_*` |
| Model comparison | `cv_scores`, `oof_mean`, `model_comparison`, `policy_fit`, `policy_coefficients` |
| Proxies and audit | `proxy_gap`, `sex_recovery`, `fairness_audit` |
| History | `policy_groups`, `child_cutoffs`, `boat_side`, `missing_age` |
| Simulation | `sim_coefficients`, `sim_gap`, `sim_recovery`, `sim_impact` |

## Limits

- **Associations, not causes.** Coefficients and rules describe the record,
  not why anyone lived or died. Predictions do not identify a historical
  decision rule and are not a guide to who "should" be selected.
- **Laplace intervals.** Intervals are approximate and marginal; there are no
  MCMC draws.
- **Forest parity.** Forest results differ between languages by design.
- **Simulation.** The synthetic data are illustrative. They show the audit can
  recover a planted bias, not how any real organisation behaves.
