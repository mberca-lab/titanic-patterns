---
title: "Titanic patterns: an R-first historical analysis"
output:
  github_document:
    toc: true
    html_preview: false
---




``` r
source("R/analysis.R")
inputs <- load_inputs(".")
d <- inputs$data
s <- inputs$split
out <- file.path("outputs", "r")
if (!file.exists(file.path(out, "metrics.csv"))) {
  run_analysis(".")
}
metrics <- read.csv(file.path(out, "metrics.csv"))
rates <- read.csv(file.path(out, "bayesian_rates.csv"))
rules <- read.csv(file.path(out, "rules.csv"))
coefficients <- read.csv(file.path(out, "coefficients.csv"))
```

## 1. A selected historical sample, not a complete manifest

The analysis uses **1309 labeled passenger records** from the pinned,
CC0-declared public source described in the root documentation. They are not
the entire Titanic manifest or a random sample of a modern population.
Names, ticket strings, and cabins are not predictors or rule items. Group
identifiers preserve shared tickets across the committed split but do not
perfectly identify families or social groups.


``` r
knitr::kable(data.frame(
  subset = c("Training", "Held-out test"),
  n = c(sum(s$subset == "train"), sum(s$subset == "test")),
  survived = c(sum(d$Survived[s$subset == "train"]),
               sum(d$Survived[s$subset == "test"]))),
  caption = "The split is committed once and shared by both languages.")
```



Table: The split is committed once and shared by both languages.

|subset        |    n| survived|
|:-------------|----:|--------:|
|Training      | 1048|      400|
|Held-out test |  261|      100|

``` r
knitr::kable(read.csv(file.path(out, "missingness.csv")), digits = 3,
             caption = paste("Missing fractions refer to all", nrow(d), "labeled records."))
```



Table: Missing fractions refer to all 1309 labeled records.

|field         | n_missing| pct_missing|
|:-------------|---------:|-----------:|
|PassengerId   |         0|       0.000|
|Survived      |         0|       0.000|
|Pclass        |         0|       0.000|
|Sex           |         0|       0.000|
|Age           |       263|       0.201|
|SibSp         |         0|       0.000|
|Parch         |         0|       0.000|
|Fare          |         1|       0.001|
|TicketSize    |         0|       0.000|
|FarePerPerson |         1|       0.001|
|Embarked      |         2|       0.002|
|Title         |         0|       0.000|
|BoatSide      |         0|       0.000|
|InKaggleTrain |         0|       0.000|

![Sample overview](overview.png)

Recorded sex and passenger class have visibly different observed survival
proportions in this selected sample. These are descriptions of historical
records, not explanations of why a person survived, causal effects, or a
basis for discriminatory decisions.

## 2. Predictive comparison: fit locally, assess on unseen groups

The three prespecified models are a training-prevalence baseline, a
500-tree random forest (`mtry=3`, `nodesize=5`), and regularized Bayesian
logistic regression. The seed is `20261014`. Each of five training validation
folds refits its own medians, embarkation mode, and sample-standard-deviation
scales. The final models fit on all 1048 training rows and transform the same
261 test rows without refitting preprocessing.

The ordered inputs are age_z, log_fare_z, family_size_z, age_missing, sex_male, class_2, class_3, embarked_Q, embarked_S. Age and fare per
person (`FarePerPerson`, the ticket fare divided by the ticket's group size)
use training medians; missing embarkation uses the alphabetical-first training
mode in a tie. Fare per person is transformed with `log1p`; family size is
`SibSp + Parch + 1`. Numeric covariates are centered and scaled with training
sample SD; zero SD is replaced by one.


``` r
cv <- metrics[metrics$phase == "cv", ]
cv_summary <- aggregate(cbind(roc_auc, brier, log_loss, balanced_accuracy) ~ model,
                        cv, mean)
knitr::kable(cv_summary, digits = 4,
             caption = "Unweighted mean of the five validation-fold scores (development only).")
```



Table: Unweighted mean of the five validation-fold scores (development only).

|model               | roc_auc|  brier| log_loss| balanced_accuracy|
|:-------------------|-------:|------:|--------:|-----------------:|
|Bayesian logistic   |  0.8293| 0.1545|   0.4814|            0.7641|
|Prevalence baseline |  0.5000| 0.2360|   0.6649|            0.5000|
|Random forest       |  0.8364| 0.1545|   0.6425|            0.7517|

These means describe the particular committed folds, not five independent
experiments or a confidence interval. No hyperparameters are selected from
the held-out scores.


``` r
test_scores <- metrics[metrics$phase == "test", ]
knitr::kable(test_scores[c("model", "roc_auc", "brier", "log_loss",
                           "balanced_accuracy", "n")], digits = 4,
             caption = "Held-out performance, reported once without test-driven tuning.")
```



Table: Held-out performance, reported once without test-driven tuning.

|   |model               | roc_auc|  brier| log_loss| balanced_accuracy|   n|
|:--|:-------------------|-------:|------:|--------:|-----------------:|---:|
|16 |Prevalence baseline |  0.5000| 0.2363|   0.6656|            0.5000| 261|
|17 |Random forest       |  0.8686| 0.1329|   0.6907|            0.8084| 261|
|18 |Bayesian logistic   |  0.8659| 0.1310|   0.4197|            0.7917| 261|

In this particular holdout, the forest's Brier score is
**0.133**,
compared with **0.236**
for the constant baseline and
**0.131**
for the Bayesian plug-in model. Thus, the modeled historical characteristics
carry predictive information in this partition; that finding does not explain
survival mechanisms or certify performance on another population.

Higher ROC AUC and balanced accuracy are better; lower Brier score and log
loss are better. Balanced accuracy uses a fixed probability threshold of
**at least 0.5**. ROC calculations always use survival as the positive event
and a fixed direction; no automatic direction selection is permitted.

![Validation and test comparison](performance.png)

![Held-out ROC curves](roc.png)

![Reliability and bin counts](calibration.png)

The test set contains only 261 records. Scores and small-bin reliability
estimates can move with another sample or grouped partition. These figures
do not establish a statistically significant gain or a universally best
algorithm. R and Python forests need not agree: their fitting algorithms
and probability definitions differ even with matched settings.

## 3. Bayesian logistic regression: approximate uncertainty, explicit prediction

The intercept has a zero-centered Normal prior with SD 5; all other
coefficients have independent zero-centered Normal priors with SD 2.5.
Analytic-gradient BFGS, followed by analytic Newton polishing, finds the
posterior mode with explicit convergence and gradient checks.
The inverse analytic negative-log-posterior Hessian
provides a **Laplace approximation** to the coefficient covariance.


``` r
knitr::kable(coefficients, digits = 4,
             caption = "Posterior modes and approximate marginal 95% credible intervals.")
```



Table: Posterior modes and approximate marginal 95% credible intervals.

|term          | estimate|     sd|   lower|   upper| odds_ratio| or_lower| or_upper| prior_sd|
|:-------------|--------:|------:|-------:|-------:|----------:|--------:|--------:|--------:|
|intercept     |   2.6349| 0.3095|  2.0283|  3.2416|    13.9424|   7.6011|  25.5740|      5.0|
|age_z         |  -0.4265| 0.0901| -0.6031| -0.2500|     0.6528|   0.5471|   0.7788|      2.5|
|log_fare_z    |   0.0297| 0.1297| -0.2246|  0.2840|     1.0302|   0.7989|   1.3285|      2.5|
|family_size_z |  -0.2410| 0.0889| -0.4152| -0.0669|     0.7858|   0.6602|   0.9353|      2.5|
|age_missing   |  -0.3040| 0.2245| -0.7441|  0.1360|     0.7378|   0.4752|   1.1457|      2.5|
|sex_male      |  -2.5828| 0.1763| -2.9283| -2.2372|     0.0756|   0.0535|   0.1068|      2.5|
|class_2       |  -0.8790| 0.2847| -1.4369| -0.3210|     0.4152|   0.2377|   0.7254|      2.5|
|class_3       |  -1.7809| 0.3215| -2.4111| -1.1507|     0.1685|   0.0897|   0.3164|      2.5|
|embarked_Q    |  -0.2815| 0.3378| -0.9436|  0.3806|     0.7547|   0.3892|   1.4632|      2.5|
|embarked_S    |  -0.5244| 0.2083| -0.9326| -0.1161|     0.5919|   0.3935|   0.8904|      2.5|

![Approximate Bayesian associations](bayesian_effects.png)

The odds ratios are conditional model associations. Numeric coefficients
represent one training sample SD; binary coefficients use female, first
class, and embarkation at C as references. The missing-age flag is included.
Marginal intervals do not capture every model, selection, or historical
measurement uncertainty. These are not frequentist confidence intervals.

All reported Bayesian predictive scores use **posterior-mode plug-in
probabilities**. They are not integrated posterior predictive probabilities.
This is neither MCMC nor an exact posterior analysis.

## 4. Exploratory association rules: count first, then interpret

Only the 1048 training records are transactions. Survival is confined to
one consequent item, never an antecedent. Antecedents use at most two items
from recorded sex, class, raw age category, family category, or raw fare-per-person
category. Missing ages/fares become `Unknown`, not imputed values. The
fixed filters are support at least 0.04, confidence at least 0.60,
joint count at least 20, lift at least 1.0, and total rule length 2–3.


``` r
knitr::kable(head(rules, 10), digits = 3,
             caption = paste(nrow(rules), "retained exploratory rules; complete export in rules.csv."))
```



Table: 45 retained exploratory rules; complete export in rules.csv.

|antecedent                       |consequent       | support| confidence|  lift| count| antecedent_count| consequent_count| baseline|    N|
|:--------------------------------|:----------------|-------:|----------:|-----:|-----:|----------------:|----------------:|--------:|----:|
|Fare=High (>30) & Sex=female     |Outcome=Survived |   0.047|      0.961| 2.517|    49|               51|              400|    0.382| 1048|
|Class=1 & Sex=female             |Outcome=Survived |   0.097|      0.953| 2.498|   102|              107|              400|    0.382| 1048|
|Fare=Middle (10-30) & Sex=female |Outcome=Survived |   0.104|      0.886| 2.322|   109|              123|              400|    0.382| 1048|
|Class=2 & Sex=female             |Outcome=Survived |   0.076|      0.879| 2.303|    80|               91|              400|    0.382| 1048|
|Age=Child (<16) & Family=2-4     |Outcome=Survived |   0.046|      0.787| 2.062|    48|               61|              400|    0.382| 1048|
|Family=2-4 & Sex=female          |Outcome=Survived |   0.139|      0.777| 2.035|   146|              188|              400|    0.382| 1048|
|Age=Adult (16-59) & Sex=female   |Outcome=Survived |   0.180|      0.762| 1.997|   189|              248|              400|    0.382| 1048|
|Family=Alone & Sex=female        |Outcome=Survived |   0.105|      0.753| 1.974|   110|              146|              400|    0.382| 1048|
|Sex=female                       |Outcome=Survived |   0.255|      0.730| 1.911|   267|              366|              400|    0.382| 1048|
|Class=1 & Family=2-4             |Outcome=Survived |   0.080|      0.700| 1.834|    84|              120|              400|    0.382| 1048|

The leading retained pattern is **Fare=High (>30) & Sex=female**
with consequent **Outcome=Survived**. It covers
**49 joint records** among **51**
matching antecedents, rather than a large lift derived from a hidden tiny
denominator. Its confidence is **96.1%**,
compared with a consequent baseline of
**38.2%** in training.

![Training-only rules](rules.png)

Support is joint count divided by the true training N. Confidence is joint
count divided by antecedent count; lift divides confidence by that rule's
consequent prevalence. Rules for non-survival therefore have a different
baseline from survival rules. Repeated and overlapping patterns are expected.
The thresholds are discovery filters, not significance tests. These
training-only exhibits are not validated predictions or causal discoveries.

## 5. A separate Bayesian question: describe sex-by-class rates

For this section **all 1309 labeled records** describe six historical strata.
A Beta(1,1) prior and `s` survivors among `n` records yield
Beta(`s+1`, `n-s+1`). The table reports the exact conjugate posterior mean
and central 95% beta credible interval. This separate descriptive
calculation does not enter model fitting, preprocessing, rule mining,
or held-out performance.


``` r
knitr::kable(rates, digits = 4,
             caption = "Full-sample descriptive posterior; group counts remain visible.")
```



Table: Full-sample descriptive posterior; group counts remain visible.

|sex    | pclass|   n| survived| observed_rate| posterior_mean|  lower|  upper| prior_alpha| prior_beta|
|:------|------:|---:|--------:|-------------:|--------------:|------:|------:|-----------:|----------:|
|female |      1| 144|      139|        0.9653|         0.9589| 0.9214| 0.9847|           1|          1|
|male   |      1| 179|       61|        0.3408|         0.3425| 0.2753| 0.4130|           1|          1|
|female |      2| 106|       94|        0.8868|         0.8796| 0.8123| 0.9337|           1|          1|
|male   |      2| 171|       25|        0.1462|         0.1503| 0.1012| 0.2070|           1|          1|
|female |      3| 216|      106|        0.4907|         0.4908| 0.4248| 0.5571|           1|          1|
|male   |      3| 493|       75|        0.1521|         0.1535| 0.1232| 0.1866|           1|          1|

![Descriptive Bayesian rates](bayesian_rates.png)

Large differences in these observed records do not identify mechanisms.
The posterior intervals are conditional on the binomial model and prior,
not a correction for missing records, selection, dependence, or confounding.

## 6. Feature comparisons and group diagnostics

`R/audit.R` mirrors `Python/audit.py`. Five Laplace-approximated Bayesian
logistic specifications (policy terms, structural terms, both, sex-blind, and
sex-blind plus title), a prevalence baseline, and a 300-tree random forest are
refitted inside each of the 20 x 5 committed grouped folds in
`data/repeated_cv.csv`. Differences from the reference *Policy + structure*
model use Nadeau-Bengio corrected resampled t intervals.

"Policy" names a demographic feature set, not an identified policy effect.
The group-average observed/predicted ratios do not establish calibration across
the probability range or fairness. The synthetic promotion analysis below is
a separate demonstration under specified assumptions, not an analogy about
historical survival or evidence about real employees.


``` r
comparison <- read.csv(file.path(out, "model_comparison.csv"))
knitr::kable(comparison[comparison$metric == "roc_auc",
                        c("model", "mean_model", "mean_difference", "lower", "upper")],
             digits = 4, caption = "Repeated grouped CV AUC differences vs Policy + structure.")
```



Table: Repeated grouped CV AUC differences vs Policy + structure.

|   |model               | mean_model| mean_difference|   lower|   upper|
|:--|:-------------------|----------:|---------------:|-------:|-------:|
|3  |Prevalence baseline |     0.5000|         -0.3552| -0.3817| -0.3287|
|6  |Policy only         |     0.7910|         -0.0641| -0.0803| -0.0480|
|9  |Structure only      |     0.6912|         -0.1640| -0.1988| -0.1291|
|12 |Policy + structure  |     0.8552|          0.0000|  0.0000|  0.0000|
|15 |Sex-blind           |     0.7058|         -0.1494| -0.1803| -0.1186|
|18 |Sex-blind + title   |     0.8466|         -0.0086| -0.0198|  0.0027|
|21 |Random forest       |     0.8489|         -0.0063| -0.0207|  0.0081|

``` r
knitr::kable(read.csv(file.path(out, "proxy_gap.csv")), digits = 3,
             caption = "How much of the observed female-male gap each model reproduces out of fold.")
```



Table: How much of the observed female-male gap each model reproduces out of fold.

|source             | female_mean| male_mean|   gap| share_of_observed_gap|
|:------------------|-----------:|---------:|-----:|---------------------:|
|Observed outcomes  |       0.727|     0.191| 0.536|                 1.000|
|Policy + structure |       0.725|     0.191| 0.533|                 0.994|
|Sex-blind          |       0.429|     0.356| 0.073|                 0.136|
|Sex-blind + title  |       0.720|     0.195| 0.525|                 0.979|

``` r
fairness <- read.csv(file.path(out, "fairness_audit.csv"))
knitr::kable(fairness[c("grouping", "group", "n", "base_rate", "selection_rate",
                        "impact_ratio", "tpr", "fpr", "ppv")], digits = 3,
             caption = "Group audit at threshold 0.5; rates with denominators below 20 are suppressed.")
```



Table: Group audit at threshold 0.5; rates with denominators below 20 are suppressed.

|grouping    |group            |   n| base_rate| selection_rate| impact_ratio|   tpr|   fpr|   ppv|
|:-----------|:----------------|---:|---------:|--------------:|------------:|-----:|-----:|-----:|
|Sex         |Female           | 466|     0.727|          0.809|        1.000| 0.900| 0.567| 0.809|
|Sex         |Male             | 843|     0.191|          0.028|        0.035| 0.124| 0.006| 0.833|
|Class       |Class 1          | 323|     0.619|          0.461|        1.000| 0.720| 0.041| 0.966|
|Class       |Class 2          | 277|     0.430|          0.426|        0.923| 0.882| 0.082| 0.890|
|Class       |Class 3          | 709|     0.255|          0.189|        0.410| 0.420| 0.110| 0.567|
|Sex x class |Female / class 1 | 144|     0.965|          1.000|        1.000| 1.000|    NA| 0.965|
|Sex x class |Female / class 2 | 106|     0.887|          1.000|        1.000| 1.000|    NA| 0.887|
|Sex x class |Female / class 3 | 216|     0.491|          0.588|        0.588| 0.679| 0.500| 0.567|
|Sex x class |Male / class 1   | 179|     0.341|          0.028|        0.028| 0.082| 0.000|    NA|
|Sex x class |Male / class 2   | 171|     0.146|          0.070|        0.070| 0.440| 0.007|    NA|
|Sex x class |Male / class 3   | 493|     0.152|          0.014|        0.014| 0.053| 0.007|    NA|

Removing recorded sex does not remove its signal when title (or other
proxies) remain. The synthetic promotion example (`sim_*.csv`) shows a model
recovering an injected gender penalty; it contains no real employees.

## Reproduce and inspect

From the repository root, run `source("R/run_analysis.R")`, then
`source("R/test-analysis.R")`. See `R/README.md` for rendering commands.
CSV files retain high precision; `preprocessing.json` records the final
training transform. `session-info.txt` records package versions and the
maximum final MAP gradient. No real-passenger simulator or rescue advice
is included.


``` r
sessionInfo()
```

```
## R version 4.6.2 (2026-10-29 ucrt)
## Platform: x86_64-w64-mingw32/x64
## Running under: Windows 11 x64 (build 26310)
## 
## Matrix products: default
##   LAPACK version 3.12.1
## 
## locale:
## [1] LC_COLLATE=English_Ireland.utf8  LC_CTYPE=English_Ireland.utf8   
## [3] LC_MONETARY=English_Ireland.utf8 LC_NUMERIC=C                    
## [5] LC_TIME=English_Ireland.utf8    
## 
## time zone: Europe/Dublin
## tzcode source: internal
## 
## attached base packages:
## [1] stats     graphics  grDevices utils     datasets  methods   base     
## 
## loaded via a namespace (and not attached):
## [1] compiler_4.6.2 cli_3.6.6      tools_4.6.2    otel_0.2.0     knitr_1.52    
## [6] xfun_0.62      rlang_1.3.0    evaluate_1.0.5
```
