# Analysis contract

This is a historical observational case study. The question is: **what do
descriptive patterns, predictive comparisons and uncertainty tell us about
the available records, and what can they not establish?**

The work makes no causal claims. Models generate record-level predictions for
evaluation, not rescue prioritisation or judgments about individuals. The data
are the 1,309 passenger records in titanic3; crew are not included. A separate
synthetic promotion example tests selected audit methods under specified
generating assumptions; it is not evidence about real employees.

## Data

- `data/passengers.csv` is a minimal titanic3 extract with a pinned checksum.
  Names, tickets, cabins, boats, body numbers and home/destination text are
  dropped at ingest. The derived columns are defined in `data/provenance.json`.
- `FarePerPerson = Fare / TicketSize`, because fares were recorded per ticket.
  All models and fare bins use this per-person value.
- `Title` is grouped as Mr, Mrs (including Mme), Miss (including Mlle and Ms),
  Master and Rare. It is used **only** in the proxy experiments.
- `BoatSide` maps each recorded boat to starboard (odd numbers, A, C) or port
  (even numbers, B, D). It is descriptive only and is never a predictor.
- **Regression test:** the 891-row Kaggle sample (`InKaggleTrain`) matches
  titanic3 exactly on survival, sex and class. `prepare_data.py` fails if this
  match breaks.

## Splits

- `data/split.csv`: one stratified, ticket-grouped holdout of 1,048 training and
  261 test rows, plus five inner folds. It feeds the original lenses (rules,
  Bayesian rates, the holdout ROC and calibration).
- `data/repeated_cv.csv`: 20 repeats × 5 stratified, ticket-grouped folds over
  all rows (seeds 20261114–20261133). It feeds every model comparison and the
  audit.
- A ticket group never spans two folds. Every preprocessing statistic is
  re-learned from the training rows of each fold:
  - medians;
  - the embarkation mode, with ties broken alphabetically;
  - means and sample standard deviations, where a zero SD is replaced by 1.

## Feature-set comparison models

The exported "Policy" labels name demographic feature sets, not an identified
decision policy or its causal effect. Feature-set comparisons assess prediction;
they do not separate the causes of survival.

Every model is a Bayesian logistic regression fitted by Laplace approximation:

- priors are Normal(0, 5²) on the intercept and Normal(0, 2.5²) on each slope;
- the posterior mode is found by Newton iteration to a maximum absolute gradient
  of at most 1e-10;
- the covariance is the inverse Hessian;
- predictions use the posterior-mode plug-in.

`child` is defined as recorded age under 16. Unknown ages are coded as not a
child and flagged separately by `age_missing`.

| Specification | Terms |
|---|---|
| Policy only | sex_male, child, male×child, age_missing |
| Structure only | class_2, class_3, log per-person fare (z), embarked Q/S, family size (z) |
| **Policy + structure** (reference) | both sets, plus male×class_2, male×class_3 and child×class_3 |
| Sex-blind | the reference model without any sex term |
| Sex-blind + title | sex-blind, plus title dummies (Mr is the reference level) |
| Random forest | benchmark only: 300 trees, 3 features per split, minimum leaf size 5 |

The differences between models are paired by (repeat, fold) and summarised with
the **Nadeau–Bengio corrected resampled t interval**:

- SE = √((1/J + 1/(K−1)) · var(d)), with J = 100 and K = 5;
- t interval with 99 degrees of freedom.

The metrics are log loss, Brier score and ROC AUC. A difference whose interval
includes zero is reported as "no reliable difference."

## Proxy tests

- **Proxy gap:** the female mean minus the male mean of the out-of-fold
  probabilities, as a share of the observed gap.
- **Sex recovery:** a Laplace logistic model predicts recorded sex from the
  non-sex features, once without and once with titles. It is scored by AUC
  across the same 100 folds.

## Fairness audit

The audit uses the reference model's out-of-fold probability, averaged over
the 20 repeats, with a prespecified threshold of 0.5. Groups are sex, class and
sex × class. For each group it reports:

- the base rate, the mean prediction and the calibration ratio;
- the selection rate, with a Wilson 95% interval;
- the **impact ratio** relative to the highest-rate group, with a Katz log
  interval, for both labels and predictions;
- TPR, FPR and PPV, with Wilson intervals.

Any rate with a denominator below **20** is suppressed.

The observed-rate / mean-prediction ratio checks agreement in group averages,
not calibration across the probability range. The four-fifths heuristic is
employment context, not a fairness test for historical survival records.
Formal incompatibility results depend on specified definitions and assumptions
and are not established by agreement in group averages. The identity
FPR = p/(1−p) · (1−PPV)/PPV · TPR is unit-tested.

## Descriptive history

- **Recorded groups:** survival for women and children vs men, by class.
- **Child cutoffs:** the descriptive groups recomputed with cutoffs at 12, 14, 16
  and 18.
- **Boat side:** the share of adult men among those boarded on each side.
- **Missing age:** the share of missing ages by class.

All proportions carry Wilson 95% intervals. Context from the British Wreck
Commissioner's Inquiry (1,178 lifeboat seats; about 711 saved) is quoted, not
computed.

## Synthetic promotion simulation

`scripts/prepare_data.py` generates 5,000 synthetic employees with seed
20261014.

**Generating process**

- merit ~ N(0, 1);
- rating = clip(3 + 0.8·(merit + N(0, 0.6²)), 1, 5);
- tenure ~ Gamma(2.2, 2.4);
- P(part-time) = 0.30 for women and 0.04 for men.

**Promotion rule (log-odds)**

−1.6 + 1.3·merit + 0.15·tenure_z − **0.70·woman** − **0.50·regional**.
Part-time status has no effect.

**Merit-only counterfactual:** the same uniform draws are reused with both
penalties removed. The truth is stored in `data/synthetic_truth.json`.

**Audit**

The simulation is split 70/30 into training and test rows, and the analysis
reports:

- the coefficients of a gender-visible and a gender-blind model, next to the
  injected truth;
- the predicted gender gap on the test rows;
- how well gender can be recovered from the remaining columns (AUC);
- impact ratios by gender and site, for both the historical and the merit-only
  labels.

**Expected result**

- The gender-visible model's woman coefficient is negative, and its interval
  excludes zero.
- The gender-blind model loads a spurious penalty onto part-time work.

The woman coefficient can be attenuated, because the rating is a noisy measure
of merit and logistic regression is non-collapsible.

## Original lenses (appendix)

These lenses use the same holdout split as before and per-person fare.

- **Rules:** Apriori on training rows only. Survival is a singleton consequent.
  The thresholds are support 0.04, confidence 0.60, joint count 20 and lift 1.0.
  These are exploratory filters, not significance tests.
- **Rates:** Beta(1,1) sex × class posterior rates, with counts shown.
- **Prediction:** a prevalence baseline, a 500-tree random forest and a
  Bayesian logistic model, fitted on the 1,048 training rows and evaluated once
  on the 261 test rows.
- **Diagnostics:** ROC curves and right-closed five-bin reliability curves
  (`[0,0.2]`, `(0.2,0.4]`, …) with bin counts.

## Reproducibility and limits

- **Seed and parity.** The seed is `20261014`. Both languages consume the
  committed splits and folds. `scripts/check_parity.py` compares every
  deterministic table between R and Python. The random forests are excluded.
- **Coverage.** The source is a curated, incomplete record: passengers only,
  with sex recorded as a binary and missing ages concentrated in third class.
- **Interpretation.** The patterns describe associations in the available
  records; they do not identify policy enforcement, causal mechanisms or any
  individual's fate. The synthetic promotion example is a separate
  demonstration, not a direct analogy or evidence about real organizations.
