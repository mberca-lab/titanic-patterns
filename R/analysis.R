FEATURES <- c("age_z", "log_fare_z", "family_size_z", "age_missing",
              "sex_male", "class_2", "class_3", "embarked_Q", "embarked_S")
NUMERIC_FEATURES <- FEATURES[1:3]
SEED <- 20261014L
MODELS <- c("Prevalence baseline", "Random forest", "Bayesian logistic")
# Committed titanic3 extract and grouped split (see data/provenance.json).
N_RECORDS <- 1309L
N_TRAIN <- 1048L
N_TEST <- 261L

check_packages <- function() {
  required <- c("arules", "randomForest", "pROC", "jsonlite", "ggplot2")
  missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) stop("Missing packages: ", paste(missing, collapse = ", "),
                            ". Run source('R/install.R').")
}

load_inputs <- function(root = ".") {
  d <- read.csv(file.path(root, "data", "passengers.csv"),
                na.strings = c("", "NA"), stringsAsFactors = FALSE)
  s <- read.csv(file.path(root, "data", "split.csv"), stringsAsFactors = FALSE)
  stopifnot(nrow(d) == N_RECORDS, !anyDuplicated(d$PassengerId),
            !anyDuplicated(s$PassengerId), setequal(d$PassengerId, s$PassengerId))
  s <- s[match(d$PassengerId, s$PassengerId), ]
  stopifnot(all(s$subset %in% c("train", "test")),
            sum(s$subset == "train") == N_TRAIN, sum(s$subset == "test") == N_TEST,
            all(d$Survived %in% 0:1),
            all(s$cv_fold[s$subset == "train"] %in% 0:4),
            setequal(s$cv_fold[s$subset == "train"], 0:4))
  validate_groups(s)
  list(data = d, split = s)
}

validate_groups <- function(s) {
  if (any(vapply(split(s$subset, s$ticket_group),
                 function(x) length(unique(x)) != 1L, logical(1))))
    stop("Ticket group crosses train/test boundary.")
  train <- s[s$subset == "train", ]
  if (any(vapply(split(train$cv_fold, train$ticket_group),
                 function(x) length(unique(x)) != 1L, logical(1))))
    stop("Ticket group crosses validation folds.")
  invisible(TRUE)
}

numeric_values <- function(d, age_median, fare_median) {
  age <- ifelse(is.na(d$Age), age_median, d$Age)
  # Fare is recorded per ticket; FarePerPerson divides it by the ticket's group size.
  fare <- ifelse(is.na(d$FarePerPerson), fare_median, d$FarePerPerson)
  if (any(!is.finite(age)) || any(!is.finite(fare)) || any(fare < 0))
    stop("Age/fare cannot be transformed.")
  cbind(age_z = age, log_fare_z = log1p(fare),
        family_size_z = d$SibSp + d$Parch + 1)
}

fit_preprocessing <- function(d) {
  age <- median(d$Age, na.rm = TRUE)
  fare <- median(d$FarePerPerson, na.rm = TRUE)
  embark <- d$Embarked[!is.na(d$Embarked) & nzchar(d$Embarked)]
  if (!length(embark)) stop("Training embarkation values are all missing.")
  frequencies <- table(embark)
  mode <- sort(names(frequencies)[frequencies == max(frequencies)], method = "radix")[1]
  x <- numeric_values(d, age, fare)
  means <- colMeans(x)
  scales <- apply(x, 2, sd)
  scales[scales == 0] <- 1
  if (any(!is.finite(scales))) stop("Training scales are not finite.")
  list(age_median = age, fare_median = fare, embarked_mode = mode,
       means = means, scales = scales, features = FEATURES)
}

transform_features <- function(d, prep) {
  numeric <- numeric_values(d, prep$age_median, prep$fare_median)
  numeric <- sweep(sweep(numeric, 2, prep$means, "-"), 2, prep$scales, "/")
  embark <- d$Embarked
  embark[is.na(embark) | !nzchar(embark)] <- prep$embarked_mode
  x <- cbind(numeric, age_missing = as.integer(is.na(d$Age)),
             sex_male = as.integer(d$Sex == "male"),
             class_2 = as.integer(d$Pclass == 2),
             class_3 = as.integer(d$Pclass == 3),
             embarked_Q = as.integer(embark == "Q"),
             embarked_S = as.integer(embark == "S"))
  storage.mode(x) <- "double"
  x[, FEATURES, drop = FALSE]
}

softplus <- function(x) pmax(x, 0) + log1p(exp(-abs(x)))

posterior_functions <- function(x, y, prior_sd) {
  precision <- 1 / prior_sd^2
  list(
    value = function(beta) {
      eta <- drop(x %*% beta)
      sum(softplus(eta) - y * eta) + sum(beta^2 * precision) / 2
    },
    gradient = function(beta) {
      drop(crossprod(x, plogis(drop(x %*% beta)) - y)) + beta * precision
    },
    hessian = function(beta) {
      p <- plogis(drop(x %*% beta))
      crossprod(x, x * (p * (1 - p))) + diag(precision, nrow = ncol(x))
    })
}

fit_bayesian <- function(x, y) {
  design <- cbind(intercept = 1, x)
  prior_sd <- c(5, rep(2.5, ncol(x)))
  objective <- posterior_functions(design, y, prior_sd)
  fit <- optim(rep(0, ncol(design)), objective$value, objective$gradient,
               method = "BFGS", control = list(maxit = 5000, reltol = 1e-13))
  if (fit$convergence != 0L)
    stop("BFGS posterior optimization failed: convergence=", fit$convergence)
  # BFGS may stop on relative objective change before the gradient is small.
  # Analytic Newton polishing makes the MAP check insensitive to that stopping rule.
  for (iteration in seq_len(10L)) {
    g <- objective$gradient(fit$par)
    if (max(abs(g)) <= 1e-8) break
    step <- solve(objective$hessian(fit$par), g)
    fit$par <- fit$par - drop(step)
  }
  gradient <- max(abs(objective$gradient(fit$par)))
  if (fit$convergence != 0L || !is.finite(gradient) || gradient > 1e-5)
    stop("Posterior optimization failed: convergence=", fit$convergence,
         "; maximum absolute gradient=", format(gradient))
  hessian <- objective$hessian(fit$par)
  chol(hessian)
  covariance <- solve(hessian)
  names(fit$par) <- colnames(design)
  list(beta = fit$par, covariance = covariance, prior_sd = prior_sd,
       gradient = gradient, convergence = fit$convergence)
}

predict_bayesian <- function(fit, x) {
  plogis(drop(cbind(intercept = 1, x) %*% fit$beta))
}

coefficient_table <- function(fit) {
  estimate <- unname(fit$beta)
  se <- sqrt(diag(fit$covariance))
  lower <- estimate - qnorm(.975) * se
  upper <- estimate + qnorm(.975) * se
  data.frame(term = names(fit$beta), estimate = estimate, sd = unname(se),
             lower = lower, upper = upper, odds_ratio = exp(estimate),
             or_lower = exp(lower), or_upper = exp(upper),
             prior_sd = fit$prior_sd)
}

score_predictions <- function(y, probability) {
  stopifnot(length(y) == length(probability), all(y %in% 0:1),
            all(is.finite(probability)), all(probability >= 0 & probability <= 1),
            length(unique(y)) == 2L)
  # sklearn >= 1.5 log_loss clips at machine epsilon; match it exactly.
  eps <- .Machine$double.eps
  clipped <- pmin(pmax(probability, eps), 1 - eps)
  positive <- probability >= .5
  c(roc_auc = as.numeric(pROC::auc(pROC::roc(y, probability, levels = c(0, 1),
                                           direction = "<", quiet = TRUE))),
    brier = mean((probability - y)^2),
    log_loss = -mean(y * log(clipped) + (1 - y) * log1p(-clipped)),
    balanced_accuracy = (mean(positive[y == 1]) + mean(!positive[y == 0])) / 2,
    n = length(y))
}

fit_models <- function(train, assessment) {
  prep <- fit_preprocessing(train)
  x_train <- transform_features(train, prep)
  x_assessment <- transform_features(assessment, prep)
  bayes <- fit_bayesian(x_train, train$Survived)
  set.seed(SEED)
  forest <- randomForest::randomForest(
    x = x_train, y = factor(train$Survived, levels = c(0, 1)),
    ntree = 500, mtry = 3, nodesize = 5)
  probabilities <- list(
    rep(mean(train$Survived), nrow(assessment)),
    unname(predict(forest, x_assessment, type = "prob")[, "1"]),
    predict_bayesian(bayes, x_assessment))
  names(probabilities) <- MODELS
  list(probabilities = probabilities, prep = prep, bayes = bayes,
       forest = forest, x_train = x_train, x_assessment = x_assessment)
}

rule_items <- function(d) {
  age <- ifelse(is.na(d$Age), "Unknown",
                ifelse(d$Age < 16, "Child (<16)",
                       ifelse(d$Age < 60, "Adult (16-59)", "Older (60+)")))
  fare <- ifelse(is.na(d$FarePerPerson), "Unknown",
                 ifelse(d$FarePerPerson <= 10, "Low (<=10)",
                        ifelse(d$FarePerPerson <= 30, "Middle (10-30)", "High (>30)")))
  family <- d$SibSp + d$Parch + 1
  family <- ifelse(family == 1, "Alone", ifelse(family <= 4, "2-4", "5+"))
  lapply(seq_len(nrow(d)), function(i) {
    c(paste0("Sex=", d$Sex[i]), paste0("Class=", d$Pclass[i]),
      paste0("Age=", age[i]), paste0("Family=", family[i]),
      paste0("Fare=", fare[i]),
      paste0("Outcome=", ifelse(d$Survived[i] == 1, "Survived", "Did not survive")))
  })
}

count_rule <- function(items, lhs, rhs) {
  left <- vapply(items, function(row) all(lhs %in% row), logical(1))
  right <- vapply(items, function(row) rhs %in% row, logical(1))
  joint <- sum(left & right)
  a <- sum(left)
  b <- sum(right)
  n <- length(items)
  data.frame(antecedent = paste(sort(lhs, method = "radix"), collapse = " & "),
             consequent = rhs, support = joint / n, confidence = joint / a,
             lift = (joint / a) / (b / n), count = joint,
             antecedent_count = a, consequent_count = b, baseline = b / n, N = n)
}

mine_rules <- function(train, support = .04, confidence = .60,
                       min_count = 20L, min_lift = 1) {
  items <- rule_items(train)
  transactions <- methods::as(items, "transactions")
  outcome <- c("Outcome=Survived", "Outcome=Did not survive")
  rules <- arules::apriori(
    transactions,
    parameter = list(supp = support, conf = confidence, minlen = 2, maxlen = 3,
                     target = "rules"),
    appearance = list(rhs = outcome, default = "lhs"),
    control = list(verbose = FALSE))
  empty <- data.frame(antecedent = character(), consequent = character(),
                      support = numeric(), confidence = numeric(), lift = numeric(),
                      count = integer(), antecedent_count = integer(),
                      consequent_count = integer(), baseline = numeric(), N = integer())
  if (!length(rules)) return(empty)
  lhs <- methods::as(arules::lhs(rules), "list")
  rhs <- methods::as(arules::rhs(rules), "list")
  rows <- lapply(seq_along(lhs), function(i) count_rule(items, lhs[[i]], rhs[[i]][1]))
  result <- do.call(rbind, rows)
  result <- result[result$count >= min_count & result$lift >= min_lift &
                     result$support >= support & result$confidence >= confidence, ]
  result <- result[order(-result$lift, -result$count, result$antecedent,
                         result$consequent, method = "radix"), ]
  rownames(result) <- NULL
  result
}

missingness_table <- function(d) {
  # Every loaded field except the joined split bookkeeping columns.
  fields <- setdiff(names(d), c("subset", "cv_fold", "ticket_group"))
  data.frame(field = fields, n_missing = colSums(is.na(d[fields])),
             pct_missing = colMeans(is.na(d[fields])), row.names = NULL)
}

bayesian_rates <- function(d) {
  groups <- expand.grid(sex = c("female", "male"), pclass = 1:3,
                        stringsAsFactors = FALSE)
  do.call(rbind, lapply(seq_len(nrow(groups)), function(i) {
    row <- d[d$Sex == groups$sex[i] & d$Pclass == groups$pclass[i], ]
    n <- nrow(row)
    survived <- sum(row$Survived)
    data.frame(sex = groups$sex[i], pclass = groups$pclass[i], n = n,
               survived = survived, observed_rate = survived / n,
               posterior_mean = (survived + 1) / (n + 2),
               lower = qbeta(.025, survived + 1, n - survived + 1),
               upper = qbeta(.975, survived + 1, n - survived + 1),
               prior_alpha = 1, prior_beta = 1)
  }))
}

write_csv <- function(x, path) {
  quote_columns <- which(vapply(x, is.character, logical(1)))
  for (name in names(x)[vapply(x, is.numeric, logical(1))]) {
    values <- x[[name]]
    x[[name]] <- sprintf("%.17g", values)
    x[[name]][is.na(values)] <- NA_character_
  }
  write.table(x, path, sep = ",", row.names = FALSE, col.names = TRUE,
              quote = quote_columns, na = "")
}

calibration_bins <- function(y, probability, model) {
  bin <- cut(probability, breaks = seq(0, 1, .2), include.lowest = TRUE)
  do.call(rbind, lapply(levels(bin), function(label) {
    use <- bin == label
    data.frame(model = model, bin = label, n = sum(use),
               predicted = if (any(use)) mean(probability[use]) else NA_real_,
               observed = if (any(use)) mean(y[use]) else NA_real_)
  }))
}

make_plots <- function(d, train, test, final, metrics, coefficients, rates, rules, out) {
  library(ggplot2)
  palette <- c("Prevalence baseline" = "#667085", "Random forest" = "#0072B2",
               "Bayesian logistic" = "#D55E00")
  theme_set(theme_minimal(base_size = 12) +
              theme(plot.title = element_text(face = "bold"),
                    panel.grid.minor = element_blank(), legend.position = "bottom"))
  save <- function(name, plot, width = 10, height = 6) {
    ggsave(file.path(out, paste0(name, ".png")), plot, width = width, height = height,
           dpi = 150, bg = "white")
  }
  overview <- aggregate(Survived ~ Sex + Pclass, d, mean)
  overview$n <- as.integer(table(factor(paste(d$Sex, d$Pclass),
                                       levels = paste(overview$Sex, overview$Pclass))))
  overview$label <- sprintf("%.0f%%\nn=%d", overview$Survived * 100, overview$n)
  save("overview", ggplot(overview, aes(factor(Pclass), Survived, fill = Sex)) +
         geom_col(position = position_dodge(.8), width = .7) +
         geom_text(aes(label = label), position = position_dodge(.8), vjust = -.3,
                   size = 3.5) +
         scale_fill_manual(values = c(female = "#0072B2", male = "#E69F00")) +
         scale_y_continuous(labels = function(x) paste0(round(x * 100), "%"),
                            limits = c(0, 1.09)) +
         labs(title = "Who is represented in these historical records?",
              subtitle = paste(format(nrow(d), big.mark = ","),
                               "titanic3 passenger records; crew excluded; descriptive, not causal"),
              x = "Passenger class", y = "Observed survival proportion", fill = "Recorded sex"))
  score_names <- c("roc_auc", "brier", "log_loss", "balanced_accuracy")
  long <- do.call(rbind, lapply(score_names, function(metric)
    data.frame(metrics[c("phase", "fold", "model")], metric = metric,
               value = metrics[[metric]])))
  long$phase <- ifelse(long$phase == "cv", "Development: five validation folds",
                       paste0("Held-out test (", nrow(test), " records)"))
  save("performance", ggplot(long, aes(model, value, color = model)) +
         geom_point(position = position_jitter(width = .10, height = 0, seed = SEED),
                    size = 2.4) +
         facet_grid(metric ~ phase, scales = "free_y") +
         scale_color_manual(values = palette) +
         labs(title = "Development and held-out scores answer different questions",
              subtitle = "Each CV point is one fold; higher AUC/balanced accuracy, lower Brier/log loss",
              x = NULL, y = "Score") +
         theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()), height = 9)
  roc_data <- do.call(rbind, lapply(MODELS, function(model) {
    roc <- pROC::roc(test$Survived, final$probabilities[[model]],
                     levels = c(0, 1), direction = "<", quiet = TRUE)
    data.frame(model = model, false_positive = 1 - roc$specificities,
               true_positive = roc$sensitivities)
  }))
  save("roc", ggplot(roc_data, aes(false_positive, true_positive, color = model)) +
         geom_abline(slope = 1, intercept = 0, linetype = 2, color = "#999999") +
         geom_path(linewidth = .8) + coord_equal() +
         scale_color_manual(values = palette) +
         labs(title = "Held-out ranking: ROC curves",
              subtitle = "Fixed positive event = survival; no automatic ROC direction selection",
              x = "False-positive rate", y = "True-positive rate"))
  calibration <- do.call(rbind, lapply(MODELS, function(model)
    calibration_bins(test$Survived, final$probabilities[[model]], model)))
  write_csv(calibration, file.path(out, "calibration.csv"))
  save("calibration", ggplot(calibration[calibration$n > 0, ],
                             aes(predicted, observed, color = model)) +
         geom_abline(slope = 1, intercept = 0, linetype = 2, color = "#999999") +
         geom_line() + geom_point(aes(size = n)) +
         geom_text(aes(label = paste0("n=", n)), vjust = -1, size = 3,
                   show.legend = FALSE) +
         scale_color_manual(values = palette) + coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
         labs(title = "Held-out reliability: five fixed probability bins",
              subtitle = "Labels expose small bin counts; an empty bin has no plotted estimate",
              x = "Mean predicted survival probability", y = "Observed survival proportion",
              size = "Bin count"))
  effects <- coefficients[coefficients$term != "intercept", ]
  effects$term <- factor(effects$term, levels = rev(FEATURES))
  save("bayesian_effects", ggplot(effects, aes(odds_ratio, term)) +
         geom_vline(xintercept = 1, linetype = 2, color = "#999999") +
         geom_segment(aes(x = or_lower, xend = or_upper, yend = term),
                      linewidth = .8, color = "#0072B2") +
         geom_point(size = 2.6, color = "#0072B2") + scale_x_log10() +
         labs(title = "Regularized associations, with approximate posterior uncertainty",
              subtitle = "Training fit: Laplace marginal 95% credible intervals; not causal effects",
              x = "Odds ratio (log scale); numeric features per training sample SD", y = NULL))
  rates$group <- paste(rates$sex, "class", rates$pclass)
  save("bayesian_rates", ggplot(rates, aes(posterior_mean, reorder(group, posterior_mean))) +
         geom_segment(aes(x = lower, xend = upper, yend = reorder(group, posterior_mean)),
                      linewidth = 1, color = "#0072B2") +
         geom_point(size = 3, color = "#0072B2") +
         geom_text(aes(label = paste0("n=", n)), nudge_y = .25, size = 3) +
         xlim(0, 1) +
         labs(title = "Descriptive Bayesian rates: sex by passenger class",
              subtitle = paste("All", format(nrow(d), big.mark = ","),
                               "labeled records; Beta(1,1) prior; exact central 95% beta credible intervals"),
              x = "Survival proportion posterior", y = NULL))
  if (nrow(rules)) {
    top <- head(rules, 10)
    top$label <- paste(top$antecedent, "=>", top$consequent)
    top$label <- factor(top$label, levels = rev(top$label))
    save("rules", ggplot(top, aes(lift, label)) +
           geom_segment(aes(x = 1, xend = lift, yend = label), color = "#999999") +
           geom_point(aes(size = count, color = confidence)) +
           scale_color_gradient(low = "#56B4E9", high = "#0072B2", limits = c(.6, 1)) +
           labs(title = "Training-only exploratory patterns: ten highest-lift rules",
                subtitle = paste0(nrow(train), " training records; support >= .04, confidence >= .60, count >= 20"),
                x = "Lift relative to the consequent's training prevalence", y = NULL,
                size = "Joint count", color = "Confidence"), width = 13, height = 7)
  }
  invisible(calibration)
}

run_analysis <- function(root = ".") {
  check_packages()
  inputs <- load_inputs(root)
  d <- inputs$data
  s <- inputs$split
  train <- d[s$subset == "train", ]
  test <- d[s$subset == "test", ]
  folds <- s$cv_fold[s$subset == "train"]
  out <- file.path(root, "outputs", "r")
  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  rows <- list()
  cv_predictions <- list()
  for (fold in 0:4) {
    assessment <- train[folds == fold, ]
    fitted <- fit_models(train[folds != fold, ], assessment)
    for (model in MODELS) {
      scores <- score_predictions(assessment$Survived, fitted$probabilities[[model]])
      rows[[length(rows) + 1L]] <- data.frame(
        phase = "cv", fold = fold, model = model, as.list(scores), check.names = FALSE)
      cv_predictions[[length(cv_predictions) + 1L]] <- data.frame(
        PassengerId = assessment$PassengerId, Survived = assessment$Survived, model = model,
        probability = fitted$probabilities[[model]], fold = fold)
    }
  }
  final <- fit_models(train, test)
  for (model in MODELS) {
    scores <- score_predictions(test$Survived, final$probabilities[[model]])
    rows[[length(rows) + 1L]] <- data.frame(
      phase = "test", fold = -1L, model = model, as.list(scores), check.names = FALSE)
  }
  metrics <- do.call(rbind, rows)
  predictions <- do.call(rbind, lapply(MODELS, function(model)
    data.frame(PassengerId = test$PassengerId, Survived = test$Survived,
               model = model, probability = final$probabilities[[model]])))
  coefficients <- coefficient_table(final$bayes)
  rules <- mine_rules(train)
  rates <- bayesian_rates(d)
  missingness <- missingness_table(d)
  tables <- list(metrics = metrics, predictions = predictions, coefficients = coefficients,
                 rules = rules, bayesian_rates = rates, missingness = missingness,
                 cv_predictions = do.call(rbind, cv_predictions),
                 design_train = data.frame(PassengerId = train$PassengerId, final$x_train),
                 design_test = data.frame(PassengerId = test$PassengerId, final$x_assessment))
  for (name in names(tables)) write_csv(tables[[name]], file.path(out, paste0(name, ".csv")))
  prep_json <- final$prep
  prep_json$means <- as.list(prep_json$means)
  prep_json$scales <- as.list(prep_json$scales)
  jsonlite::write_json(prep_json, file.path(out, "preprocessing.json"),
                       auto_unbox = TRUE, pretty = TRUE, digits = 17)
  make_plots(d, train, test, final, metrics, coefficients, rates, rules, out)
  if (!exists("run_audit", mode = "function"))
    sys.source(file.path(root, "R", "audit.R"), envir = environment(run_analysis))
  started <- proc.time()[["elapsed"]]
  audit <- run_audit(d, root, out)
  tables <- c(tables, audit)
  message(sprintf("R audit complete in %.1f s: %d audit tables.",
                  proc.time()[["elapsed"]] - started, length(audit)))
  writeLines(c(capture.output(sessionInfo()),
               paste("Seed:", SEED),
               paste("Final MAP maximum absolute gradient:", format(final$bayes$gradient, digits = 17)),
               "Prediction: posterior-mode plug-in; covariance: inverse analytic Hessian."),
             file.path(out, "session-info.txt"))
  message("R analysis complete: ", nrow(train), " train / ", nrow(test),
          " test; ", nrow(rules), " exploratory rules. Outputs: ", out)
  print(metrics[metrics$phase == "test", ], row.names = FALSE)
  invisible(list(data = d, split = s, train = train, test = test,
                 final = final, tables = tables))
}
