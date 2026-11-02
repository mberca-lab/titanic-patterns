# What a model learns from historical decisions: a people-analytics audit.
# R edition of Python/audit.py. Every estimate is deterministic given the committed
# data and folds and is checked against the Python outputs by scripts/check_parity.py.
# The random forest is a flexible benchmark only and is not expected to match Python.
# Requires R/analysis.R (posterior_functions, softplus, write_csv, SEED).

TITLES <- c("title_Mrs", "title_Miss", "title_Master", "title_Rare")
POLICY <- c("sex_male", "child", "male_x_child", "age_missing")
STRUCTURE <- c("class_2", "class_3", "log_fare_pp_z", "embarked_Q", "embarked_S", "family_size_z")
INTERACTIONS <- c("male_x_class_2", "male_x_class_3", "child_x_class_3")
SEX_TERMS <- c("sex_male", "male_x_child", "male_x_class_2", "male_x_class_3")
COMBINED <- c(POLICY, STRUCTURE, INTERACTIONS)
SEX_BLIND <- setdiff(COMBINED, SEX_TERMS)
SPECS <- list(
  "Policy only" = POLICY,
  "Structure only" = STRUCTURE,
  "Policy + structure" = COMBINED,
  "Sex-blind" = SEX_BLIND,
  "Sex-blind + title" = c(SEX_BLIND, TITLES))
FOREST_FEATURES <- c("sex_male", "age_z", "age_missing", "class_2", "class_3",
                     "log_fare_pp_z", "embarked_Q", "embarked_S", "family_size_z")
SEX_RECOVERY <- list("Without title" = c("age_z", "age_missing", "class_2", "class_3",
                                         "log_fare_pp_z", "embarked_Q", "embarked_S",
                                         "sibsp_z", "parch_z"))
SEX_RECOVERY[["With title"]] <- c(SEX_RECOVERY[["Without title"]], TITLES)
REFERENCE <- "Policy + structure"
AUDIT_MODELS <- c("Prevalence baseline", names(SPECS), "Random forest")
MIN_DENOMINATOR <- 20
CHILD_AGE <- 16
Z975 <- qnorm(.975)

# Python-compatible ordering of labels (code-point order, like sorted()).
py_sort <- function(x) sort(unique(x), method = "radix")

audit_raw <- function(d, prep) {
  cbind(age_z = ifelse(is.na(d$Age), prep$age_median, d$Age),
        log_fare_pp_z = log1p(ifelse(is.na(d$FarePerPerson), prep$fare_median, d$FarePerPerson)),
        family_size_z = as.double(d$SibSp + d$Parch + 1),
        sibsp_z = as.double(d$SibSp), parch_z = as.double(d$Parch))
}

# Training-only statistics; every transform reuses them unchanged.
audit_prep <- function(train) {
  embark <- train$Embarked[!is.na(train$Embarked)]
  if (!length(embark) || all(is.na(train$Age)) || all(is.na(train$FarePerPerson)))
    stop("Training data cannot supply imputation statistics.")
  frequencies <- table(embark)
  prep <- list(age_median = median(train$Age, na.rm = TRUE),
               fare_median = median(train$FarePerPerson, na.rm = TRUE),
               embarked_mode = sort(names(frequencies)[frequencies == max(frequencies)],
                                    method = "radix")[1])
  raw <- audit_raw(train, prep)
  prep$means <- colMeans(raw)
  scales <- apply(raw, 2, sd)
  scales[scales == 0] <- 1
  prep$scales <- scales
  prep
}

audit_transform <- function(d, prep) {
  x <- sweep(sweep(audit_raw(d, prep), 2, prep$means, "-"), 2, prep$scales, "/")
  male <- as.double(d$Sex == "male")
  child <- as.double(!is.na(d$Age) & d$Age < CHILD_AGE)  # unknown age -> 0, flagged separately
  embarked <- ifelse(is.na(d$Embarked), prep$embarked_mode, d$Embarked)
  class_2 <- as.double(d$Pclass == 2)
  class_3 <- as.double(d$Pclass == 3)
  x <- cbind(x, sex_male = male, child = child, age_missing = as.double(is.na(d$Age)),
             male_x_child = male * child, class_2 = class_2, class_3 = class_3,
             embarked_Q = as.double(embarked == "Q"), embarked_S = as.double(embarked == "S"),
             male_x_class_2 = male * class_2, male_x_class_3 = male * class_3,
             child_x_class_3 = child * class_3)
  titles <- vapply(TITLES, function(t) as.double(!is.na(d$Title) & d$Title == sub("^title_", "", t)),
                   numeric(nrow(d)))
  x <- cbind(x, matrix(titles, nrow = nrow(d), dimnames = list(NULL, TITLES)))
  rownames(x) <- NULL
  x
}

# Normal(0, 5^2) intercept / Normal(0, 2.5^2) slopes; posterior mode + inverse Hessian.
laplace_fit <- function(x, y) {
  x <- as.matrix(x)
  design <- cbind(intercept = rep(1, nrow(x)), x)
  y <- as.double(y)
  prior_sd <- c(5, rep(2.5, ncol(design) - 1L))
  objective <- posterior_functions(design, y, prior_sd)
  theta <- rep(0, ncol(design))
  for (iteration in seq_len(200L)) {
    gradient <- objective$gradient(theta)
    if (max(abs(gradient)) <= 1e-10) break
    step <- drop(solve(objective$hessian(theta), gradient))
    rate <- 1
    if (max(abs(gradient)) > 1e-6) {  # damping only far from the mode
      current <- objective$value(theta)
      while (objective$value(theta - rate * step) > current && rate > 1e-10) rate <- rate / 2
    }
    theta <- theta - rate * step
  }
  gradient <- max(abs(objective$gradient(theta)))
  if (!is.finite(gradient) || gradient > 1e-7)
    stop("Laplace mode not stationary: ", format(gradient))
  hessian <- objective$hessian(theta)
  chol(hessian)
  eta <- drop(design %*% theta)
  list(theta = theta, covariance = solve(hessian), terms = colnames(design),
       prior_sd = prior_sd, gradient = gradient, loglik = sum(y * eta - softplus(eta)))
}

laplace_predict <- function(fit, x) {
  plogis(drop(cbind(rep(1, nrow(x)), as.matrix(x)) %*% fit$theta))
}

audit_coefficient_table <- function(fit, model) {
  sd <- sqrt(diag(fit$covariance))
  data.frame(model = model, term = fit$terms, estimate = fit$theta, sd = unname(sd),
             lower = fit$theta - Z975 * sd, upper = fit$theta + Z975 * sd,
             odds_ratio = exp(fit$theta), row.names = NULL)
}

# Rank (Mann-Whitney) ROC AUC with averaged ties; event 1 positive, no direction reversal.
auc <- function(y, p) {
  positive <- y == 1
  n1 <- sum(positive)
  n0 <- length(y) - n1
  (sum(rank(p)[positive]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

fold_scores <- function(y, p) {
  y <- as.double(y)
  p <- pmin(pmax(as.double(p), 1e-15), 1 - 1e-15)
  c(roc_auc = if (mean(y) > 0 && mean(y) < 1) auc(y, p) else NA_real_,
    brier = mean((p - y)^2),
    log_loss = -mean(y * log(p) + (1 - y) * log1p(-p)),
    n = length(y))
}

wilson <- function(successes, n, z = Z975) {
  if (n == 0) return(c(NA_real_, NA_real_))
  p <- successes / n
  centre <- (p + z * z / (2 * n)) / (1 + z * z / n)
  half <- z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
  c(centre - half, centre + half)
}

# Ratio of two proportions (group / reference) with a Katz log interval.
katz_ratio <- function(x1, n1, x2, n2, z = Z975) {
  if (min(x1, x2, n1, n2) == 0) {
    ratio <- if (n1 != 0 && n2 != 0 && x2 != 0) (x1 / n1) / (x2 / n2) else NA_real_
    return(c(ratio, NA_real_, NA_real_))
  }
  ratio <- (x1 / n1) / (x2 / n2)
  se <- sqrt(1 / x1 - 1 / n1 + 1 / x2 - 1 / n2)
  c(ratio, ratio * exp(-z * se), ratio * exp(z * se))
}

# Corrected resampled t interval for repeated K-fold paired differences.
nadeau_bengio <- function(differences, test_train_ratio) {
  d <- as.double(differences)
  j <- length(d)
  se <- sqrt((1 / j + test_train_ratio) * var(d))
  q <- qt(.975, j - 1)
  c(mean = mean(d), sd = sd(d), se = se, lower = mean(d) - q * se, upper = mean(d) + q * se)
}

# Fit every model inside each of the repeated grouped folds; scores and OOF probabilities.
repeated_cv <- function(d, folds, forest_trees = 300L) {
  repeats <- sort(unique(folds[["repeat"]]))
  if (!length(repeats)) stop("Repeated folds need a 'repeat' column.")
  k <- max(folds$fold) + 1L
  models <- c("Prevalence baseline", names(SPECS), "Random forest")
  score_rows <- list()
  oof_sum <- matrix(0, nrow(d), length(models), dimnames = list(NULL, models))
  oof_count <- integer(nrow(d))
  first_seen <- integer(0)
  add_score <- function(task, r, fold, model, s) {
    score_rows[[length(score_rows) + 1L]] <<- data.frame(
      task = task, "repeat" = r, fold = fold, model = model, roc_auc = s[["roc_auc"]],
      brier = s[["brier"]], log_loss = s[["log_loss"]], n = s[["n"]], check.names = FALSE)
  }
  for (r in repeats) {
    current <- folds[folds[["repeat"]] == r, ]
    fold_of <- current$fold[match(d$PassengerId, current$PassengerId)]
    stopifnot(!anyNA(fold_of))
    for (fold in seq_len(k) - 1L) {
      held <- fold_of == fold
      train <- d[!held, ]
      test <- d[held, ]
      prep <- audit_prep(train)
      x_train <- audit_transform(train, prep)
      x_test <- audit_transform(test, prep)
      probabilities <- list("Prevalence baseline" = rep(mean(train$Survived), nrow(test)))
      for (model in names(SPECS)) {
        features <- SPECS[[model]]
        fit <- laplace_fit(x_train[, features, drop = FALSE], train$Survived)
        probabilities[[model]] <- laplace_predict(fit, x_test[, features, drop = FALSE])
      }
      set.seed(SEED + 1000L * r + fold)
      forest <- randomForest::randomForest(
        x = x_train[, FOREST_FEATURES], y = factor(train$Survived, levels = c(0, 1)),
        ntree = forest_trees, mtry = 3, nodesize = 5)
      probabilities[["Random forest"]] <- unname(
        predict(forest, x_test[, FOREST_FEATURES], type = "prob")[, "1"])
      for (task in names(SEX_RECOVERY)) {
        features <- SEX_RECOVERY[[task]]
        fit <- laplace_fit(x_train[, features, drop = FALSE], x_train[, "sex_male"])
        p <- laplace_predict(fit, x_test[, features, drop = FALSE])
        add_score("Recover recorded sex", r, fold, task, fold_scores(x_test[, "sex_male"], p))
      }
      for (model in models) {
        add_score("Predict survival", r, fold, model, fold_scores(test$Survived, probabilities[[model]]))
        oof_sum[held, model] <- oof_sum[held, model] + probabilities[[model]]
      }
      oof_count[held] <- oof_count[held] + 1L
      first_seen <- c(first_seen, setdiff(which(held), first_seen))
    }
  }
  # Python groupby(sort=False) order: models, then passengers by first appearance.
  oof_mean <- do.call(rbind, lapply(models, function(model)
    data.frame(PassengerId = d$PassengerId[first_seen], model = model,
               probability = (oof_sum[, model] / oof_count)[first_seen])))
  list(scores = do.call(rbind, score_rows), oof_mean = oof_mean)
}

compare_models <- function(scores, k) {
  survival <- scores[scores$task == "Predict survival", ]
  reference <- survival[survival$model == REFERENCE, ]
  key <- function(x) paste(x[["repeat"]], x$fold)
  rows <- list()
  for (model in AUDIT_MODELS) {
    current <- survival[survival$model == model, ]
    matched <- reference[match(key(current), key(reference)), ]
    for (metric in c("log_loss", "brier", "roc_auc")) {
      nb <- nadeau_bengio(current[[metric]] - matched[[metric]], 1 / (k - 1))
      rows[[length(rows) + 1L]] <- data.frame(
        model = model, metric = metric, reference = REFERENCE,
        mean_model = mean(current[[metric]]), sd_model = sd(current[[metric]]),
        mean_reference = mean(reference[[metric]]),
        mean_difference = nb[["mean"]], sd_difference = nb[["sd"]],
        corrected_se = nb[["se"]], lower = nb[["lower"]], upper = nb[["upper"]],
        folds = nrow(current))
    }
  }
  do.call(rbind, rows)
}

sex_recovery <- function(scores) {
  sex <- scores[scores$task == "Recover recorded sex", ]
  do.call(rbind, lapply(unique(sex$model), function(model) {
    auc <- sex$roc_auc[sex$model == model]
    auc <- auc[!is.na(auc)]
    data.frame(features = model, mean_auc = mean(auc), sd_auc = sd(auc),
               min_auc = min(auc), max_auc = max(auc), folds = length(auc))
  }))
}

proxy_gap <- function(d, oof_mean) {
  female <- d$Sex == "female"
  rows <- list(data.frame(source = "Observed outcomes",
                          female_mean = mean(d$Survived[female]),
                          male_mean = mean(d$Survived[!female])))
  for (model in c("Policy + structure", "Sex-blind", "Sex-blind + title")) {
    current <- oof_mean[oof_mean$model == model, ]
    is_female <- female[match(current$PassengerId, d$PassengerId)]
    rows[[length(rows) + 1L]] <- data.frame(
      source = model, female_mean = mean(current$probability[is_female]),
      male_mean = mean(current$probability[!is_female]))
  }
  table <- do.call(rbind, rows)
  observed <- mean(d$Survived[female]) - mean(d$Survived[!female])
  table$gap <- table$female_mean - table$male_mean
  table$share_of_observed_gap <- table$gap / observed
  table
}

capitalize <- function(x) paste0(toupper(substring(x, 1, 1)), tolower(substring(x, 2)))

group_labels <- function(d) {
  list("Sex" = capitalize(d$Sex),
       "Class" = paste("Class", d$Pclass),
       "Sex x class" = paste0(capitalize(d$Sex), " / class ", d$Pclass))
}

# max over (rate, label): highest rate, ties broken by the largest label (Python tuple order).
python_argmax <- function(rates, labels) {
  best <- which(rates == max(rates))
  py_sort(labels[best])[length(best)]
}

# HR-style group audit on out-of-fold probabilities; small denominators suppressed.
fairness_audit <- function(d, probability, threshold = .5, model = REFERENCE) {
  y <- d$Survived
  p <- as.double(probability)
  selected <- p >= threshold
  rows <- list()
  for (grouping in names(group_labels(d))) {
    labels <- group_labels(d)[[grouping]]
    groups <- py_sort(labels)
    stats <- lapply(groups, function(g) {
      m <- labels == g
      list(n = sum(m), positives = sum(y[m]), selected = sum(selected[m]),
           tp = sum((selected & y == 1)[m]), fp = sum((selected & y == 0)[m]),
           mean_predicted = mean(p[m]))
    })
    names(stats) <- groups
    get <- function(field) vapply(stats, function(s) as.double(s[[field]]), numeric(1))
    reference_label <- python_argmax(get("positives") / get("n"), groups)
    reference_select <- python_argmax(get("selected") / get("n"), groups)
    for (g in groups) {
      s <- stats[[g]]
      n <- s$n
      pos <- s$positives
      neg <- n - pos
      sel <- s$selected
      selection <- wilson(sel, n)
      r <- stats[[reference_label]]
      label_ratio <- katz_ratio(pos, n, r$positives, r$n)
      r <- stats[[reference_select]]
      select_ratio <- katz_ratio(sel, n, r$selected, r$n)
      row <- data.frame(
        model = model, grouping = grouping, group = g, n = n, base_rate = pos / n,
        mean_predicted = s$mean_predicted, calibration_ratio = (pos / n) / s$mean_predicted,
        selection_rate = sel / n, selection_lower = selection[1], selection_upper = selection[2],
        label_reference = reference_label, label_impact_ratio = label_ratio[1],
        label_ir_lower = label_ratio[2], label_ir_upper = label_ratio[3],
        selection_reference = reference_select, impact_ratio = select_ratio[1],
        ir_lower = select_ratio[2], ir_upper = select_ratio[3])
      for (spec in list(list("tpr", s$tp, pos), list("fpr", s$fp, neg), list("ppv", s$tp, sel))) {
        name <- spec[[1]]
        k <- spec[[2]]
        denominator <- spec[[3]]
        ok <- denominator >= MIN_DENOMINATOR
        interval <- if (ok) wilson(k, denominator) else c(NA_real_, NA_real_)
        row[[name]] <- if (ok) k / denominator else NA_real_
        row[[paste0(name, "_lower")]] <- interval[1]
        row[[paste0(name, "_upper")]] <- interval[2]
        row[[paste0(name, "_denominator")]] <- denominator
      }
      rows[[length(rows) + 1L]] <- row
    }
  }
  do.call(rbind, rows)
}

rate_values <- function(survived, n) {
  interval <- wilson(survived, n)
  data.frame(n = n, survived = survived, rate = if (n) survived / n else NA_real_,
             lower = interval[1], upper = interval[2])
}

history_tables <- function(d) {
  known <- !is.na(d$Age)
  child <- known & d$Age < CHILD_AGE
  protected <- d$Sex == "female" | child
  classes <- c(list(All = rep(TRUE, nrow(d))),
               setNames(lapply(1:3, function(c) d$Pclass == c), as.character(1:3)))
  policy <- do.call(rbind, lapply(names(classes), function(c) {
    mask <- classes[[c]]
    do.call(rbind, lapply(list(list("Women and children", protected), list("Men", !protected)),
                          function(g) {
      m <- g[[2]] & mask
      cbind(data.frame(group = g[[1]], pclass = c), rate_values(sum(d$Survived[m]), sum(m)))
    }))
  }))
  cutoffs <- do.call(rbind, lapply(c(12, 14, 16, 18), function(cut) {
    do.call(rbind, lapply(names(classes), function(c) {
      m <- known & d$Age < cut & classes[[c]]
      m[is.na(m)] <- FALSE
      cbind(data.frame(cutoff = cut, pclass = c), rate_values(sum(d$Survived[m]), sum(m)))
    }))
  }))
  boarded <- d[d$BoatSide != "None recorded", ]
  adult_men <- boarded$Sex == "male" & boarded$Title != "Master" &
    !(!is.na(boarded$Age) & boarded$Age < CHILD_AGE)
  sides <- do.call(rbind, lapply(c("Starboard", "Port", "Unknown"), function(side) {
    m <- boarded$BoatSide == side
    v <- rate_values(sum(adult_men[m]), sum(m))
    data.frame(side = side, n = v$n, lower = v$lower, upper = v$upper,
               adult_men = v$survived, adult_men_share = v$rate)
  }))
  missing <- do.call(rbind, lapply(names(classes), function(c) {
    mask <- classes[[c]]
    v <- rate_values(sum(is.na(d$Age[mask])), sum(mask))
    data.frame(pclass = c, n = v$n, lower = v$lower, upper = v$upper,
               missing_age = v$survived, missing_share = v$rate)
  }))
  list(policy_groups = policy, child_cutoffs = cutoffs, boat_side = sides, missing_age = missing)
}

promotion_design <- function(frame, stats) {
  cbind(rating_z = (frame$rating - stats$rating_mean) / stats$rating_sd,
        tenure_z = (frame$tenure_years - stats$tenure_mean) / stats$tenure_sd,
        woman = as.double(frame$gender == "Woman"),
        regional = as.double(frame$site == "Regional"),
        part_time = as.double(frame$part_time))
}

simulation_audit <- function(promotions, truth) {
  train <- promotions[promotions$subset == "train", ]
  test <- promotions[promotions$subset == "test", ]
  stats <- list(rating_mean = mean(train$rating), rating_sd = sd(train$rating),
                tenure_mean = mean(train$tenure_years), tenure_sd = sd(train$tenure_years))
  x_train <- promotion_design(train, stats)
  x_test <- promotion_design(test, stats)
  specs <- list("Gender visible" = c("rating_z", "tenure_z", "woman", "regional", "part_time"),
                "Gender-blind" = c("rating_z", "tenure_z", "regional", "part_time"))
  injected <- unlist(truth$injected_log_odds)
  woman <- test$gender == "Woman"
  outcomes <- c("Historical decisions" = "promoted", "Merit-only counterfactual" = "promoted_merit_only")
  gaps <- lapply(names(outcomes), function(label)
    data.frame(source = label, women_mean = mean(test[[outcomes[[label]]]][woman]),
               men_mean = mean(test[[outcomes[[label]]]][!woman])))
  coefficients <- list()
  for (model in names(specs)) {
    features <- specs[[model]]
    fit <- laplace_fit(x_train[, features, drop = FALSE], train$promoted)
    table <- audit_coefficient_table(fit, model)
    table$injected <- unname(injected[table$term])
    coefficients[[model]] <- table
    p <- laplace_predict(fit, x_test[, features, drop = FALSE])
    gaps[[length(gaps) + 1L]] <- data.frame(source = model, women_mean = mean(p[woman]),
                                            men_mean = mean(p[!woman]))
  }
  gaps <- do.call(rbind, gaps)
  gaps$gap <- gaps$women_mean - gaps$men_mean
  recovery_features <- c("rating_z", "tenure_z", "regional", "part_time")
  fit <- laplace_fit(x_train[, recovery_features], x_train[, "woman"])
  recovery <- data.frame(target = "Woman", features = "Rating, tenure, site, part-time",
                         test_auc = auc(x_test[, "woman"], laplace_predict(fit, x_test[, recovery_features])),
                         test_n = nrow(test))
  impact <- list()
  for (label in names(outcomes)) {
    for (grouping in c("Gender", "Site")) {
      field <- c(Gender = "gender", Site = "site")[[grouping]]
      groups <- py_sort(promotions[[field]])
      sums <- vapply(groups, function(g) sum(promotions[[outcomes[[label]]]][promotions[[field]] == g]), numeric(1))
      counts <- vapply(groups, function(g) sum(promotions[[field]] == g), numeric(1))
      reference <- groups[which.max(sums / counts)]  # idxmax: first maximum in sorted order
      for (i in seq_along(groups)) {
        interval <- wilson(sums[[i]], counts[[i]])
        ratio <- katz_ratio(sums[[i]], counts[[i]], sums[[reference]], counts[[reference]])
        impact[[length(impact) + 1L]] <- data.frame(
          label = label, grouping = grouping, group = groups[i], n = counts[[i]],
          promoted = sums[[i]], rate = sums[[i]] / counts[[i]], lower = interval[1],
          upper = interval[2], reference = reference, impact_ratio = ratio[1],
          ir_lower = ratio[2], ir_upper = ratio[3])
      }
    }
  }
  list(sim_coefficients = do.call(rbind, unname(coefficients)), sim_gap = gaps,
       sim_recovery = recovery, sim_impact = do.call(rbind, impact))
}

run_audit <- function(d, root = ".", out = file.path(root, "outputs", "r"), forest_trees = 300L) {
  folds <- read.csv(file.path(root, "data", "repeated_cv.csv"), check.names = FALSE)
  if (!setequal(folds$PassengerId, d$PassengerId))
    stop("Repeated folds must cover every passenger")
  k <- max(folds$fold) + 1L
  cv <- repeated_cv(d, folds, forest_trees)
  full <- audit_transform(d, audit_prep(d))
  fits <- lapply(SPECS, function(features) laplace_fit(full[, features, drop = FALSE], d$Survived))
  null_loglik <- laplace_fit(full[, character(0), drop = FALSE], d$Survived)$loglik
  policy_fit <- data.frame(model = names(SPECS), terms = lengths(SPECS),
                           loglik = vapply(fits, `[[`, numeric(1), "loglik"), row.names = NULL)
  policy_fit$mcfadden_r2 <- 1 - policy_fit$loglik / null_loglik
  oof_reference <- cv$oof_mean[cv$oof_mean$model == REFERENCE, ]
  reference <- oof_reference$probability[match(d$PassengerId, oof_reference$PassengerId)]
  promotions <- read.csv(file.path(root, "data", "synthetic_promotions.csv"), stringsAsFactors = FALSE)
  truth <- jsonlite::fromJSON(file.path(root, "data", "synthetic_truth.json"))
  tables <- c(list(
    cv_scores = cv$scores, oof_mean = cv$oof_mean,
    model_comparison = compare_models(cv$scores, k), sex_recovery = sex_recovery(cv$scores),
    policy_fit = policy_fit,
    policy_coefficients = do.call(rbind, unname(Map(audit_coefficient_table, fits, names(fits)))),
    proxy_gap = proxy_gap(d, cv$oof_mean),
    fairness_audit = fairness_audit(d, reference)),
    history_tables(d), simulation_audit(promotions, truth))
  for (name in names(tables)) {
    rownames(tables[[name]]) <- NULL
    write_csv(tables[[name]], file.path(out, paste0(name, ".csv")))
  }
  tables
}
