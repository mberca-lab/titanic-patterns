# R-first historical analysis

Open `titanic-patterns.Rproj` in RStudio, or run R from the repository root.
The R analysis is self-contained and never reads Python outputs. It uses only
the committed public `data/passengers.csv`, grouped `data/split.csv`, repeated
grouped folds `data/repeated_cv.csv`, and the synthetic `data/synthetic_promotions.csv`.
Read [the shared analysis contract](../METHODOLOGY.md) before interpreting results.

## Run

```r
source("R/install.R")         # Installs missing packages only; no upgrades.
source("R/run_analysis.R")    # Complete analysis, CSVs, and static PNGs.
source("R/test-analysis.R")   # Focused unit + committed-data integration tests.
source("R/render_analysis.R") # Execute the narrative and save Markdown.
```

On Windows, the equivalent commands are:

```powershell
& 'C:\Program Files\R\R-4.6.2\bin\Rscript.exe' R\run_analysis.R
& 'C:\Program Files\R\R-4.6.2\bin\Rscript.exe' R\test-analysis.R
& 'C:\Program Files\R\R-4.6.2\bin\Rscript.exe' R\render_analysis.R
.\.venv\Scripts\python.exe scripts\check_parity.py   # R vs Python parity (after both runs)
```

`Rscript` need not be on `PATH`; call it by its full path as above. The full
run (main analysis plus the 20 x 5 repeated-CV audit) takes about 30 seconds.

All commands run from the repository root. No download, Python, Shiny server,
LaTeX, or Pandoc is needed to execute the analysis. Rendering uses
`github_document` when Pandoc is available; otherwise `knitr::knit` executes
all R Markdown chunks directly. Both routes persist `outputs/r/analysis.md`,
with relative links to the neighboring PNGs. No standalone HTML is generated.

## Files and responsibilities

| File | Purpose |
| --- | --- |
| `analysis.R` | Reusable preprocessing, models, scoring, rules, beta updates, plots |
| `audit.R` | People-analytics audit mirroring `Python/audit.py`; called by `run_analysis()` |
| `run_analysis.R` | Root-relative executable entry point |
| `test-analysis.R` | Hand-calculated metrics, leakage/group separation, analytic derivative and known posterior fixture tests, rule recounts, export integration |
| `analysis.Rmd` | Runnable narrative with computed tables and uncertainty limitations |
| `render_analysis.R` | Execute the narrative, gracefully avoiding a Pandoc dependency |
| `install.R` | Missing-package-only installation |

## Outputs

All artifacts are regenerated under `outputs/r/`:

- `metrics.csv`: each model on five CV folds and one test subset (18 rows).
  `fold=-1` marks the test subset; CV folds are 0–4.
- `predictions.csv`: held-out records only, one probability per model.
- `coefficients.csv`: intercept, then the nine ordered features; mode,
  Laplace SD, approximate central 95% credible interval, odds ratios and
  corresponding intervals, and prior SD.
- `design_train.csv`, `design_test.csv`: IDs followed by the shared nine-feature
  matrix in original data order. IDs are audit keys, never predictors.
- `preprocessing.json`: full-training medians, alphabetical-tie mode,
  numeric means/sample-SD scales, and ordered feature names.
- `rules.csv`: the complete retained training-only rule set, with raw joint,
  antecedent, and consequent counts, true training denominator, and baseline.
- `bayesian_rates.csv`: full-sample sex-by-class Beta(1,1) posterior summaries.
- `missingness.csv`: count and fraction missing for every retained field.
- `calibration.csv`: all five fixed bins per test model; empty bins are
  explicitly retained with count zero and unavailable estimates.
- Seven PNGs: `overview`, `performance`, `roc`, `calibration`,
  `bayesian_effects`, `bayesian_rates`, and `rules`.
- `cv_predictions.csv`: development-fold probabilities for every model.
- Executed `analysis.md` and `session-info.txt`.

Audit tables (`audit.R`, same names and columns as `outputs/python/`):

- `cv_scores.csv`, `oof_mean.csv`: 20 x 5 grouped repeated-CV scores (survival and
  recorded-sex recovery) and out-of-fold mean probabilities per model.
- `model_comparison.csv`: differences from *Policy + structure* with Nadeau-Bengio
  corrected intervals; `sex_recovery.csv`: how well proxies recover recorded sex.
- `policy_fit.csv`, `policy_coefficients.csv`: full-data Laplace fits of the five specs.
- `proxy_gap.csv`, `fairness_audit.csv`: out-of-fold sex gap reproduction; group
  selection rates, Wilson intervals, Katz impact ratios, TPR/FPR/PPV (suppressed when
  the denominator is below 20).
- `policy_groups.csv`, `child_cutoffs.csv`, `boat_side.csv`, `missing_age.csv`:
  historical descriptive tables.
- `sim_coefficients.csv`, `sim_gap.csv`, `sim_recovery.csv`, `sim_impact.csv`:
  synthetic promotion audit (no real employees).

All Laplace-model tables match Python within 2e-6 (most within 1e-12). Repeated-CV
random-forest rows use `randomForest` and are not expected to match scikit-learn.
A few fold ROC AUCs differ by exactly one tied pair: identical design rows have equal
probabilities in R, while numpy's matmul can separate them by one ulp;
`scripts/check_parity.py` verifies and reports each such flip.

The CSV writer uses 17-digit precision; all predictors and preprocessing
values remain unrounded. No fitted model or personal passenger simulator
is serialized.

## Interpretation guardrails

This is the 1,309-record titanic3 passenger list (crew excluded), not the complete manifest.
Preprocessing and Apriori see training records only; every CV fold relearns
its transform. Forest settings are fixed at 500 trees, `mtry=3`, `nodesize=5`,
factor outcome levels `0,1`, positive probability `"1"`, seed `20261014`.

The logistic model uses independent Normal priors (intercept SD 5; others
2.5), explicit analytic-gradient BFGS checks with analytic Newton polishing
to a stationary mode, and an inverse analytic Hessian Laplace covariance.
Predictive probabilities are **MAP plug-in estimates**,
not integrated or exact posterior predictions. Intervals are approximate
marginal credible intervals, not confidence intervals.

Descriptive Beta(1,1) rates deliberately use the full sample and are clearly
separate from the holdout exercise. Association rules are exploratory,
overlapping training patterns, not significance tests. Shared features
and Bayesian modes should closely match the Python edition; forests need
not. No causal claim, rescue advice, universal algorithm ranking, or
individual decision-use is justified.
