source(file.path("R", "analysis.R"))
source(file.path("R", "audit.R"))
check_packages()
if (!requireNamespace("testthat", quietly = TRUE)) stop("testthat is required.")
library(testthat)

test_that("metrics retain event direction, threshold ties, and stable losses", {
  y <- c(0, 0, 1, 1)
  p <- c(.1, .6, .5, .9)
  score <- score_predictions(y, p)
  expect_equal(unname(score["roc_auc"]), .75)
  expect_equal(unname(score["brier"]), .1575)
  expect_equal(unname(score["balanced_accuracy"]), .75)
  expect_equal(unname(score["log_loss"]), -mean(log(c(.9, .4, .5, .9))))
  reverse <- score_predictions(y, c(.9, .8, .2, .1))
  expect_equal(unname(reverse["roc_auc"]), 0)
  constant <- score_predictions(y, rep(.5, 4))
  expect_equal(unname(constant[c("roc_auc", "balanced_accuracy")]), c(.5, .5))
  expect_true(is.finite(score_predictions(y, c(0, 1, 0, 1))["log_loss"]))
})

fixture <- data.frame(
  PassengerId = 1:4, Survived = c(0, 1, 0, 1), Pclass = c(1, 2, 3, 1),
  Sex = c("female", "male", "male", "female"),
  Age = c(10, 20, NA, 30), SibSp = c(0, 1, 0, 0), Parch = c(0, 0, 0, 1),
  Fare = c(10, 30, NA, 20), FarePerPerson = c(10, 30, NA, 20),
  Embarked = c("S", "C", NA, NA))

test_that("preprocessing learns only fit rows, with sample SD and explicit missingness", {
  prep <- fit_preprocessing(fixture)
  expect_equal(prep$age_median, 20)
  expect_equal(prep$fare_median, 20)
  expect_equal(prep$embarked_mode, "C")
  expect_equal(unname(prep$scales["age_z"]), sd(c(10, 20, 20, 30)))
  x <- transform_features(fixture, prep)
  expect_equal(colnames(x), FEATURES)
  expect_equal(unname(x[3, "age_z"]), 0)
  expect_equal(unname(x[3, "age_missing"]), 1)
  expect_equal(unname(x[3, c("embarked_Q", "embarked_S")]), c(0, 0))
  expect_equal(unname(colMeans(x[, 1:3])), rep(0, 3), tolerance = 1e-14)
  assessment <- fixture
  assessment$Age <- c(9999, 9998, NA, 9997)
  before <- prep
  assessed <- transform_features(assessment, prep)
  expect_identical(prep, before)
  expect_equal(unname(assessed[3, "age_z"]), 0)
  expect_gt(assessed[1, "age_z"], 100)
  constant <- fixture
  constant$Age <- 20
  constant$Fare <- constant$FarePerPerson <- 10
  constant$SibSp <- constant$Parch <- 0
  expect_equal(unname(fit_preprocessing(constant)$scales), rep(1, 3))
})

test_that("committed split separates groups across holdout and all validation folds", {
  inputs <- load_inputs(".")
  expect_true(validate_groups(inputs$split))
  bad <- data.frame(subset = c("train", "test"), cv_fold = c(0, -1),
                    ticket_group = c("same", "same"))
  expect_error(validate_groups(bad), "train/test")
  bad$subset <- "train"
  bad$cv_fold <- c(0, 1)
  expect_error(validate_groups(bad), "validation")
})

test_that("analytic posterior derivatives match finite differences", {
  x <- cbind(1, c(-1.2, .4, 1.5, -.7), c(.1, -.2, .6, .3))
  y <- c(0, 1, 1, 0)
  prior <- c(5, 2.5, 2.5)
  objective <- posterior_functions(x, y, prior)
  beta <- c(.2, -.4, .7)
  eps <- 1e-5
  shifts <- diag(eps, length(beta))
  numeric_gradient <- vapply(seq_along(beta), function(i)
    (objective$value(beta + shifts[, i]) - objective$value(beta - shifts[, i])) /
      (2 * eps), numeric(1))
  numeric_hessian <- vapply(seq_along(beta), function(i)
    (objective$gradient(beta + shifts[, i]) - objective$gradient(beta - shifts[, i])) /
      (2 * eps), numeric(length(beta)))
  expect_equal(objective$gradient(beta), numeric_gradient, tolerance = 1e-8)
  expect_equal(objective$hessian(beta), numeric_hessian, tolerance = 1e-8)
  expect_true(all(eigen(objective$hessian(beta), symmetric = TRUE)$values > 0))
})

test_that("balanced intercept-only fixture has known MAP and Laplace covariance", {
  x <- matrix(numeric(0), nrow = 4, ncol = 0)
  fit <- fit_bayesian(x, c(0, 1, 0, 1))
  expect_equal(unname(fit$beta), 0, tolerance = 1e-10)
  expect_equal(unname(fit$covariance[1, 1]), 1 / (1 + 1 / 25), tolerance = 1e-10)
  expect_equal(predict_bayesian(fit, x), rep(.5, 4))
  objective <- posterior_functions(matrix(1, 4, 1), c(0, 1, 0, 1), 5)
  expect_true(is.finite(objective$value(1000)))
  expect_true(is.finite(objective$value(-1000)))
})

test_that("rule counts and categories use raw records and never put outcome on the left", {
  items <- rule_items(fixture)
  expect_true("Age=Unknown" %in% items[[3]])
  expect_true("Fare=Unknown" %in% items[[3]])
  counted <- count_rule(items, "Sex=female", "Outcome=Survived")
  expect_equal(counted$count, 1)
  expect_equal(counted$antecedent_count, 2)
  expect_equal(counted$consequent_count, 2)
  expect_equal(counted$support, .25)
  expect_equal(counted$confidence, .5)
  expect_equal(counted$lift, 1)
  inputs <- load_inputs(".")
  train <- inputs$data[inputs$split$subset == "train", ]
  rules <- mine_rules(train)
  expect_gt(nrow(rules), 0)
  expect_false(any(grepl("Outcome=", rules$antecedent, fixed = TRUE)))
  expect_true(all(rules$consequent %in% c("Outcome=Survived", "Outcome=Did not survive")))
  expect_true(all(rules$N == nrow(train)))
  expect_true(all(rules$count >= 20))
  for (i in seq_len(nrow(rules))) {
    lhs <- strsplit(rules$antecedent[i], " & ", fixed = TRUE)[[1]]
    recounted <- count_rule(rule_items(train), lhs, rules$consequent[i])
    actual <- rules[i, ]
    rownames(actual) <- NULL
    expect_equal(actual, recounted)
  }
})

test_that("descriptive beta posterior uses exact conjugate updates", {
  inputs <- load_inputs(".")
  rates <- bayesian_rates(inputs$data)
  expect_equal(sum(rates$n), N_RECORDS)
  expect_equal(sum(rates$survived), sum(inputs$data$Survived))
  expect_equal(rates$posterior_mean, (rates$survived + 1) / (rates$n + 2))
  expect_equal(rates$lower, qbeta(.025, rates$survived + 1, rates$n - rates$survived + 1))
})

test_that("pipeline exports complete held-out schemas and ordered designs", {
  out <- file.path("outputs", "r")
  expect_true(file.exists(file.path(out, "metrics.csv")))
  metrics <- read.csv(file.path(out, "metrics.csv"))
  predictions <- read.csv(file.path(out, "predictions.csv"))
  coefficients <- read.csv(file.path(out, "coefficients.csv"))
  expect_equal(names(metrics), c("phase", "fold", "model", "roc_auc", "brier",
                                  "log_loss", "balanced_accuracy", "n"))
  expect_equal(nrow(metrics), 18)
  expect_equal(sort(unique(metrics$fold)), -1:4)
  expect_equal(nrow(predictions), N_TEST * 3)
  expect_equal(names(predictions), c("PassengerId", "Survived", "model", "probability"))
  expect_true(all(table(predictions$PassengerId) == 3))
  inputs <- load_inputs(".")
  expect_setequal(unique(predictions$PassengerId),
                  inputs$data$PassengerId[inputs$split$subset == "test"])
  expect_equal(coefficients$term, c("intercept", FEATURES))
  for (subset in c("train", "test")) {
    design <- read.csv(file.path(out, paste0("design_", subset, ".csv")))
    expect_equal(names(design), c("PassengerId", FEATURES))
    expect_equal(design$PassengerId, inputs$data$PassengerId[inputs$split$subset == subset])
    expect_true(all(is.finite(as.matrix(design[-1]))))
  }
})

test_that("CSV numerical exports retain seventeen significant digits", {
  path <- file.path("outputs", "r", "precision-check.csv")
  on.exit(unlink(path), add = TRUE)
  write_csv(data.frame(value = c(pi, 1 / 3)), path)
  expect_equal(readLines(path)[-1], sprintf("%.17g", c(pi, 1 / 3)))
  expect_equal(read.csv(path)$value, c(pi, 1 / 3), tolerance = 1e-16)
})
test_that("Nadeau-Bengio interval inflates the naive resampling standard error", {
  d <- c(.01, .03, -.02, .04, .02, .00, .01, .05, -.01, .03)
  nb <- nadeau_bengio(d, 1 / 4)
  expect_equal(unname(nb["se"]), sqrt((1 / 10 + 1 / 4) * var(d)))
  expect_gt(unname(nb["se"]), sd(d) / sqrt(length(d)))
  q <- qt(.975, 9)
  expect_equal(unname(nb[c("lower", "upper")]), mean(d) + c(-1, 1) * q * unname(nb["se"]))
})

test_that("Wilson and Katz intervals handle zero counts like the Python edition", {
  interval <- wilson(0, 10)
  expect_equal(interval[1], 0, tolerance = 1e-15)
  expect_true(interval[2] > 0 && interval[2] < .35)
  expect_true(all(is.na(wilson(0, 0))))
  ratio <- katz_ratio(20, 100, 40, 100)
  expect_equal(ratio[1], .5)
  expect_true(ratio[2] < .5 && .5 < ratio[3])
  expect_equal(katz_ratio(0, 10, 5, 10)[1], 0)
  expect_true(all(is.na(katz_ratio(0, 10, 5, 10)[2:3])))
})

test_that("audit preprocessing learns training statistics only and codes unknown age as non-child", {
  inputs <- load_inputs(".")
  train <- inputs$data[inputs$split$subset == "train", ]
  prep <- audit_prep(train)
  shifted <- inputs$data
  shifted$Age[inputs$split$subset == "test"] <- 1e6
  before <- prep
  x <- audit_transform(shifted, prep)
  expect_identical(prep, before)
  expect_equal(prep$age_median, median(train$Age, na.rm = TRUE))
  expect_true(all(x[is.na(shifted$Age), "child"] == 0))
  expect_true(all(unlist(SPECS) %in% colnames(x)))
})

test_that("fairness audit suppresses small denominators and satisfies the Chouldechova identity", {
  inputs <- load_inputs(".")
  set.seed(3)
  audit <- fairness_audit(inputs$data, runif(nrow(inputs$data)))
  for (name in c("tpr", "fpr", "ppv")) {
    small <- audit[[paste0(name, "_denominator")]] < MIN_DENOMINATOR
    expect_true(all(is.na(audit[small, c(name, paste0(name, c("_lower", "_upper")))])))
    expect_false(anyNA(audit[[name]][!small]))
  }
  full <- audit[!is.na(audit$tpr) & !is.na(audit$fpr) & !is.na(audit$ppv) &
                  audit$ppv > 0 & audit$ppv < 1, ]
  expect_gt(nrow(full), 0)
  p <- full$base_rate
  expect_equal(full$fpr, p / (1 - p) * (1 - full$ppv) / full$ppv * full$tpr, tolerance = 1e-12)
  committed <- read.csv(file.path("outputs", "r", "fairness_audit.csv"))
  full <- committed[!is.na(committed$tpr) & !is.na(committed$fpr) & !is.na(committed$ppv) &
                      committed$ppv > 0 & committed$ppv < 1, ]
  p <- full$base_rate
  expect_equal(full$fpr, p / (1 - p) * (1 - full$ppv) / full$ppv * full$tpr, tolerance = 1e-12)
  expect_true(all(is.na(committed$ppv[committed$ppv_denominator < MIN_DENOMINATOR])))
})

test_that("exported audit reproduces the documented findings", {
  out <- file.path("outputs", "r")
  folds <- read.csv(file.path("data", "repeated_cv.csv"), check.names = FALSE)
  expect_true(all(table(folds[["repeat"]]) == N_RECORDS))
  comparison <- read.csv(file.path(out, "model_comparison.csv"))
  expect_true(all(comparison$folds == 100))
  scores <- read.csv(file.path(out, "cv_scores.csv"), check.names = FALSE)
  expect_equal(names(scores), c("task", "repeat", "fold", "model", "roc_auc", "brier",
                                "log_loss", "n"))
  coefficients <- read.csv(file.path(out, "sim_coefficients.csv"))
  woman <- coefficients[coefficients$model == "Gender visible" & coefficients$term == "woman", ]
  truth <- jsonlite::fromJSON(file.path("data", "synthetic_truth.json"))
  expect_lt(woman$upper, 0)
  expect_true(woman$lower < truth$injected_log_odds$woman && truth$injected_log_odds$woman < woman$upper)
  fit <- read.csv(file.path(out, "policy_fit.csv"))
  expect_gt(fit$loglik[fit$model == "Policy + structure"], max(fit$loglik[fit$model %in% c("Policy only", "Structure only")]))
})
cat("\nAll focused R tests passed.\n")
