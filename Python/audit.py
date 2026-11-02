"""What a model learns from historical decisions: a people-analytics audit.

Every estimate here is deterministic given the committed data and folds, so the R
edition (R/audit.R) must reproduce it. The random forest is a flexible benchmark only;
its probabilities are not expected to match across languages.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm, t as student_t
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from .analysis import ROOT, SEED, posterior_objective

TITLES = ["title_Mrs", "title_Miss", "title_Master", "title_Rare"]
POLICY = ["sex_male", "child", "male_x_child", "age_missing"]
STRUCTURE = ["class_2", "class_3", "log_fare_pp_z", "embarked_Q", "embarked_S", "family_size_z"]
INTERACTIONS = ["male_x_class_2", "male_x_class_3", "child_x_class_3"]
SEX_TERMS = ["sex_male", "male_x_child", "male_x_class_2", "male_x_class_3"]
COMBINED = POLICY + STRUCTURE + INTERACTIONS
SEX_BLIND = [c for c in COMBINED if c not in SEX_TERMS]
SPECS = {
    "Policy only": POLICY,
    "Structure only": STRUCTURE,
    "Policy + structure": COMBINED,
    "Sex-blind": SEX_BLIND,
    "Sex-blind + title": SEX_BLIND + TITLES,
}
FOREST_FEATURES = ["sex_male", "age_z", "age_missing", "class_2", "class_3",
                   "log_fare_pp_z", "embarked_Q", "embarked_S", "family_size_z"]
SEX_RECOVERY = {
    "Without title": ["age_z", "age_missing", "class_2", "class_3", "log_fare_pp_z",
                      "embarked_Q", "embarked_S", "sibsp_z", "parch_z"],
}
SEX_RECOVERY["With title"] = SEX_RECOVERY["Without title"] + TITLES
REFERENCE = "Policy + structure"
AUDIT_MODELS = ["Prevalence baseline"] + list(SPECS) + ["Random forest"]
MIN_DENOMINATOR = 20
CHILD_AGE = 16


class AuditPrep:
    """Training-only statistics; every transform reuses them unchanged."""

    def __init__(self, train):
        self.age_median = float(train.Age.median())
        self.fare_median = float(train.FarePerPerson.median())
        modes = train.Embarked.dropna().mode().sort_values()
        self.embarked_mode = str(modes.iloc[0])
        raw = self._raw(train)
        self.means = {c: float(raw[c].mean()) for c in raw}
        self.scales = {c: float(raw[c].std(ddof=1)) or 1.0 for c in raw}

    def _raw(self, data):
        return pd.DataFrame({
            "age_z": data.Age.fillna(self.age_median),
            "log_fare_pp_z": np.log1p(data.FarePerPerson.fillna(self.fare_median)),
            "family_size_z": (data.SibSp + data.Parch + 1).astype(float),
            "sibsp_z": data.SibSp.astype(float), "parch_z": data.Parch.astype(float),
        }, index=data.index)

    def transform(self, data):
        x = self._raw(data)
        for c in x:
            x[c] = (x[c] - self.means[c]) / self.scales[c]
        male = data.Sex.eq("male").astype(float)
        child = (data.Age < CHILD_AGE).astype(float)  # unknown age -> 0, flagged separately
        embarked = data.Embarked.fillna(self.embarked_mode)
        x["sex_male"], x["child"] = male, child
        x["age_missing"] = data.Age.isna().astype(float)
        x["male_x_child"] = male * child
        x["class_2"] = data.Pclass.eq(2).astype(float)
        x["class_3"] = data.Pclass.eq(3).astype(float)
        x["embarked_Q"] = embarked.eq("Q").astype(float)
        x["embarked_S"] = embarked.eq("S").astype(float)
        x["male_x_class_2"] = male * x.class_2
        x["male_x_class_3"] = male * x.class_3
        x["child_x_class_3"] = child * x.class_3
        for title in TITLES:
            x[title] = data.Title.eq(title.removeprefix("title_")).astype(float)
        return x.astype(float)


def laplace_fit(x, y):
    """Normal(0, 5^2) intercept / Normal(0, 2.5^2) slopes; posterior mode + inverse Hessian."""
    design = np.column_stack([np.ones(len(x)), np.asarray(x, dtype=float)])
    y = np.asarray(y, dtype=float)
    prior_sd = np.array([5.] + [2.5] * (design.shape[1] - 1))
    objective = lambda theta: posterior_objective(theta, design, y, prior_sd)
    result = minimize(lambda t: objective(t)[0], np.zeros(design.shape[1]),
                      jac=lambda t: objective(t)[1], hess=lambda t: objective(t)[2],
                      method="trust-exact", options={"gtol": 1e-10, "maxiter": 500})
    theta = result.x
    for _ in range(10):  # Newton polish to a tight stationary point
        _, gradient, hessian = objective(theta)
        if np.max(np.abs(gradient)) <= 1e-10:
            break
        theta = theta - np.linalg.solve(hessian, gradient)
    _, gradient, hessian = objective(theta)
    if np.max(np.abs(gradient)) > 1e-7:
        raise RuntimeError(f"Laplace mode not stationary: {np.max(np.abs(gradient))}")
    np.linalg.cholesky(hessian)
    terms = ["intercept"] + list(getattr(x, "columns", range(design.shape[1] - 1)))
    return {"theta": theta, "covariance": np.linalg.inv(hessian), "terms": terms,
            "prior_sd": prior_sd, "gradient": float(np.max(np.abs(gradient))),
            "loglik": float(np.sum(y * (design @ theta) - np.logaddexp(0, design @ theta)))}


def laplace_predict(fit, x):
    return expit(np.column_stack([np.ones(len(x)), np.asarray(x, dtype=float)]) @ fit["theta"])


def coefficient_table(fit, model):
    sd = np.sqrt(np.diag(fit["covariance"]))
    z = norm.ppf(.975)
    return pd.DataFrame({
        "model": model, "term": fit["terms"], "estimate": fit["theta"], "sd": sd,
        "lower": fit["theta"] - z * sd, "upper": fit["theta"] + z * sd,
        "odds_ratio": np.exp(fit["theta"]),
    })


def fold_scores(y, p):
    y, p = np.asarray(y, dtype=float), np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    return {
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else np.nan,
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))),
        "n": int(len(y)),
    }


def wilson(successes, n, z=norm.ppf(.975)):
    if n == 0:
        return np.nan, np.nan
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def katz_ratio(x1, n1, x2, n2, z=norm.ppf(.975)):
    """Ratio of two proportions (group / reference) with a Katz log interval."""
    if min(x1, x2, n1, n2) == 0:
        return (x1 / n1) / (x2 / n2) if n1 and n2 and x2 else np.nan, np.nan, np.nan
    ratio = (x1 / n1) / (x2 / n2)
    se = np.sqrt(1 / x1 - 1 / n1 + 1 / x2 - 1 / n2)
    return ratio, ratio * np.exp(-z * se), ratio * np.exp(z * se)


def nadeau_bengio(differences, test_train_ratio):
    """Corrected resampled t interval for repeated K-fold paired differences."""
    d = np.asarray(differences, dtype=float)
    j = len(d)
    se = np.sqrt((1 / j + test_train_ratio) * np.var(d, ddof=1))
    q = student_t.ppf(.975, j - 1)
    return float(d.mean()), float(np.std(d, ddof=1)), float(se), float(d.mean() - q * se), float(d.mean() + q * se)


def repeated_cv(data, folds, forest_trees=300):
    """Fit every model inside each of the 20 x 5 grouped folds; return scores and OOF probabilities."""
    scores, oof = [], []
    fold_of = folds.pivot(index="PassengerId", columns="repeat", values="fold").loc[data.PassengerId]
    for repeat in sorted(folds["repeat"].unique()):
        for fold in range(int(folds.fold.max()) + 1):
            held = fold_of[repeat].to_numpy() == fold
            train, test = data.loc[~held], data.loc[held]
            prep = AuditPrep(train)
            x_train, x_test = prep.transform(train), prep.transform(test)
            probabilities = {"Prevalence baseline": np.full(len(test), train.Survived.mean())}
            for model, features in SPECS.items():
                fit = laplace_fit(x_train[features], train.Survived)
                probabilities[model] = laplace_predict(fit, x_test[features])
            forest = RandomForestClassifier(n_estimators=forest_trees, max_features=3,
                                            min_samples_leaf=5, n_jobs=2,
                                            random_state=SEED + 1000 * repeat + fold)
            forest.fit(x_train[FOREST_FEATURES], train.Survived)
            probabilities["Random forest"] = forest.predict_proba(x_test[FOREST_FEATURES])[:, 1]
            for task, features in SEX_RECOVERY.items():
                fit = laplace_fit(x_train[features], x_train.sex_male)
                p = laplace_predict(fit, x_test[features])
                scores.append({"task": "Recover recorded sex", "repeat": repeat, "fold": fold,
                               "model": task, **fold_scores(x_test.sex_male, p)})
            for model, p in probabilities.items():
                scores.append({"task": "Predict survival", "repeat": repeat, "fold": fold,
                               "model": model, **fold_scores(test.Survived, p)})
                oof.append(pd.DataFrame({"PassengerId": test.PassengerId.to_numpy(),
                                         "repeat": repeat, "model": model, "probability": p}))
    oof = pd.concat(oof, ignore_index=True)
    oof_mean = (oof.groupby(["model", "PassengerId"], sort=False).probability.mean()
                .reset_index()[["PassengerId", "model", "probability"]])
    return pd.DataFrame(scores), oof_mean


def compare_models(scores, k):
    survival = scores.loc[scores.task.eq("Predict survival")]
    reference = survival.loc[survival.model.eq(REFERENCE)].set_index(["repeat", "fold"])
    rows = []
    for model in AUDIT_MODELS:
        current = survival.loc[survival.model.eq(model)].set_index(["repeat", "fold"])
        for metric in ["log_loss", "brier", "roc_auc"]:
            mean, sd, se, lower, upper = nadeau_bengio(
                (current[metric] - reference.loc[current.index, metric]).to_numpy(), 1 / (k - 1))
            rows.append({"model": model, "metric": metric, "reference": REFERENCE,
                         "mean_model": float(current[metric].mean()),
                         "sd_model": float(current[metric].std(ddof=1)),
                         "mean_reference": float(reference[metric].mean()),
                         "mean_difference": mean, "sd_difference": sd, "corrected_se": se,
                         "lower": lower, "upper": upper, "folds": len(current)})
    return pd.DataFrame(rows)


def sex_recovery(scores):
    sex = scores.loc[scores.task.eq("Recover recorded sex")]
    return (sex.groupby("model", sort=False).roc_auc.agg(["mean", "std", "min", "max", "count"])
            .reset_index().rename(columns={"model": "features", "mean": "mean_auc", "std": "sd_auc",
                                          "min": "min_auc", "max": "max_auc", "count": "folds"}))


def proxy_gap(data, oof_mean):
    female = data.set_index("PassengerId").Sex.eq("female")
    observed = data.loc[data.Sex.eq("female"), "Survived"].mean() - data.loc[data.Sex.eq("male"), "Survived"].mean()
    rows = [{"source": "Observed outcomes", "female_mean": data.loc[data.Sex.eq("female"), "Survived"].mean(),
             "male_mean": data.loc[data.Sex.eq("male"), "Survived"].mean()}]
    for model in ["Policy + structure", "Sex-blind", "Sex-blind + title"]:
        p = oof_mean.loc[oof_mean.model.eq(model)].set_index("PassengerId").probability
        rows.append({"source": model, "female_mean": p[female[p.index]].mean(),
                     "male_mean": p[~female[p.index]].mean()})
    table = pd.DataFrame(rows)
    table["gap"] = table.female_mean - table.male_mean
    table["share_of_observed_gap"] = table.gap / observed
    return table


def group_labels(data):
    return {
        "Sex": data.Sex.str.capitalize(),
        "Class": "Class " + data.Pclass.astype(str),
        "Sex x class": data.Sex.str.capitalize() + " / class " + data.Pclass.astype(str),
    }


def fairness_audit(data, probability, threshold=.5, model=REFERENCE):
    """HR-style group audit on out-of-fold probabilities; small denominators suppressed."""
    y = data.Survived.to_numpy()
    p = np.asarray(probability)
    selected = p >= threshold
    rows = []
    for grouping, labels in group_labels(data).items():
        labels = labels.to_numpy()
        stats = {}
        for group in sorted(set(labels)):
            m = labels == group
            stats[group] = {
                "n": int(m.sum()), "positives": int(y[m].sum()), "selected": int(selected[m].sum()),
                "tp": int((selected & (y == 1))[m].sum()), "fp": int((selected & (y == 0))[m].sum()),
                "mean_predicted": float(p[m].mean()),
            }
        reference_label = max(stats, key=lambda g: (stats[g]["positives"] / stats[g]["n"], g))
        reference_select = max(stats, key=lambda g: (stats[g]["selected"] / stats[g]["n"], g))
        for group, s in stats.items():
            n, pos, neg, sel = s["n"], s["positives"], s["n"] - s["positives"], s["selected"]
            row = {"model": model, "grouping": grouping, "group": group, "n": n,
                   "base_rate": pos / n, "mean_predicted": s["mean_predicted"],
                   "calibration_ratio": (pos / n) / s["mean_predicted"],
                   "selection_rate": sel / n}
            row["selection_lower"], row["selection_upper"] = wilson(sel, n)
            r = stats[reference_label]
            row["label_reference"] = reference_label
            row["label_impact_ratio"], row["label_ir_lower"], row["label_ir_upper"] = katz_ratio(
                pos, n, r["positives"], r["n"])
            r = stats[reference_select]
            row["selection_reference"] = reference_select
            row["impact_ratio"], row["ir_lower"], row["ir_upper"] = katz_ratio(
                sel, n, r["selected"], r["n"])
            for name, k, d in [("tpr", s["tp"], pos), ("fpr", s["fp"], neg), ("ppv", s["tp"], sel)]:
                ok = d >= MIN_DENOMINATOR
                row[name] = k / d if ok else np.nan
                row[f"{name}_lower"], row[f"{name}_upper"] = wilson(k, d) if ok else (np.nan, np.nan)
                row[f"{name}_denominator"] = d
            rows.append(row)
    return pd.DataFrame(rows)


def rate_row(**labels):
    survived, n = labels.pop("survived"), labels.pop("n")
    lower, upper = wilson(survived, n)
    return {**labels, "n": n, "survived": survived,
            "rate": survived / n if n else np.nan, "lower": lower, "upper": upper}


def history_tables(data):
    known = data.Age.notna()
    child = known & (data.Age < CHILD_AGE)
    protected = data.Sex.eq("female") | child
    classes = [("All", np.ones(len(data), bool))] + [
        (str(c), data.Pclass.eq(c).to_numpy()) for c in (1, 2, 3)]
    policy = [rate_row(group=g, pclass=c, n=int((m & mask).sum()), survived=int(data.Survived[m & mask].sum()))
              for c, mask in classes for g, m in [("Women and children", protected.to_numpy()),
                                                 ("Men", (~protected).to_numpy())]]
    cutoffs = [rate_row(cutoff=cut, pclass=c, n=int((m & mask).sum()), survived=int(data.Survived[m & mask].sum()))
               for cut in (12, 14, 16, 18) for c, mask in classes
               for m in [(known & (data.Age < cut)).to_numpy()]]
    boarded = data.loc[data.BoatSide.ne("None recorded")]
    adult_men = boarded.Sex.eq("male") & boarded.Title.ne("Master") & ~(boarded.Age < CHILD_AGE)
    sides = []
    for side in ["Starboard", "Port", "Unknown"]:
        m = boarded.BoatSide.eq(side)
        row = rate_row(side=side, n=int(m.sum()), survived=int(adult_men[m].sum()))
        row["adult_men"] = row.pop("survived")
        row["adult_men_share"] = row.pop("rate")
        sides.append(row)
    missing = [rate_row(pclass=c, n=int(mask.sum()), survived=int(data.Age.isna()[mask].sum()))
               for c, mask in classes]
    for row in missing:
        row["missing_age"] = row.pop("survived")
        row["missing_share"] = row.pop("rate")
    return {"policy_groups": pd.DataFrame(policy), "child_cutoffs": pd.DataFrame(cutoffs),
            "boat_side": pd.DataFrame(sides), "missing_age": pd.DataFrame(missing)}


def promotion_design(frame, stats):
    return pd.DataFrame({
        "rating_z": (frame.rating - stats["rating_mean"]) / stats["rating_sd"],
        "tenure_z": (frame.tenure_years - stats["tenure_mean"]) / stats["tenure_sd"],
        "woman": frame.gender.eq("Woman").astype(float),
        "regional": frame.site.eq("Regional").astype(float),
        "part_time": frame.part_time.astype(float),
    }, index=frame.index)


def simulation_audit(promotions, truth):
    train = promotions.loc[promotions.subset.eq("train")]
    test = promotions.loc[promotions.subset.eq("test")]
    stats = {"rating_mean": train.rating.mean(), "rating_sd": train.rating.std(ddof=1),
             "tenure_mean": train.tenure_years.mean(), "tenure_sd": train.tenure_years.std(ddof=1)}
    x_train, x_test = promotion_design(train, stats), promotion_design(test, stats)
    specs = {"Gender visible": ["rating_z", "tenure_z", "woman", "regional", "part_time"],
             "Gender-blind": ["rating_z", "tenure_z", "regional", "part_time"]}
    injected = truth["injected_log_odds"]
    coefficients, gaps = [], []
    woman = test.gender.eq("Woman").to_numpy()
    for label, column in [("Historical decisions", "promoted"), ("Merit-only counterfactual", "promoted_merit_only")]:
        gaps.append({"source": label, "women_mean": test[column][woman].mean(),
                     "men_mean": test[column][~woman].mean()})
    for model, features in specs.items():
        fit = laplace_fit(x_train[features], train.promoted)
        table = coefficient_table(fit, model)
        table["injected"] = table.term.map(injected)
        coefficients.append(table)
        p = laplace_predict(fit, x_test[features])
        gaps.append({"source": model, "women_mean": p[woman].mean(), "men_mean": p[~woman].mean()})
    gaps = pd.DataFrame(gaps)
    gaps["gap"] = gaps.women_mean - gaps.men_mean
    recovery_features = ["rating_z", "tenure_z", "regional", "part_time"]
    fit = laplace_fit(x_train[recovery_features], x_train.woman)
    recovery = pd.DataFrame([{"target": "Woman", "features": "Rating, tenure, site, part-time",
                              "test_auc": roc_auc_score(x_test.woman, laplace_predict(fit, x_test[recovery_features])),
                              "test_n": len(test)}])
    impact = []
    for label, column in [("Historical decisions", "promoted"), ("Merit-only counterfactual", "promoted_merit_only")]:
        for grouping, field in [("Gender", "gender"), ("Site", "site")]:
            counts = promotions.groupby(field)[column].agg(["sum", "count"])
            reference = (counts["sum"] / counts["count"]).sort_index().idxmax()
            for group, row in counts.iterrows():
                lower, upper = wilson(row["sum"], row["count"])
                ratio, ir_lower, ir_upper = katz_ratio(row["sum"], row["count"],
                                                       counts.loc[reference, "sum"], counts.loc[reference, "count"])
                impact.append({"label": label, "grouping": grouping, "group": group,
                               "n": int(row["count"]), "promoted": int(row["sum"]),
                               "rate": row["sum"] / row["count"], "lower": lower, "upper": upper,
                               "reference": reference, "impact_ratio": ratio,
                               "ir_lower": ir_lower, "ir_upper": ir_upper})
    return {"sim_coefficients": pd.concat(coefficients, ignore_index=True), "sim_gap": gaps,
            "sim_recovery": recovery, "sim_impact": pd.DataFrame(impact)}


def run_audit(data, root=ROOT, output=None, forest_trees=300):
    root = Path(root)
    output = Path(output or root / "outputs" / "python")
    folds = pd.read_csv(root / "data" / "repeated_cv.csv")
    if set(folds.PassengerId) != set(data.PassengerId):
        raise ValueError("Repeated folds must cover every passenger")
    k = int(folds.fold.max()) + 1
    scores, oof_mean = repeated_cv(data, folds, forest_trees)
    prep = AuditPrep(data)
    full = prep.transform(data)
    fits = {model: laplace_fit(full[features], data.Survived) for model, features in SPECS.items()}
    null_loglik = laplace_fit(full[[]], data.Survived)["loglik"]
    policy_fit = pd.DataFrame([{
        "model": model, "terms": len(features), "loglik": fit["loglik"],
        "mcfadden_r2": 1 - fit["loglik"] / null_loglik,
    } for (model, features), fit in zip(SPECS.items(), fits.values())])
    reference = oof_mean.loc[oof_mean.model.eq(REFERENCE)].set_index("PassengerId").probability.loc[data.PassengerId]
    promotions = pd.read_csv(root / "data" / "synthetic_promotions.csv")
    truth = json.loads((root / "data" / "synthetic_truth.json").read_text(encoding="utf-8"))
    tables = {
        "cv_scores": scores, "oof_mean": oof_mean,
        "model_comparison": compare_models(scores, k), "sex_recovery": sex_recovery(scores),
        "policy_fit": policy_fit,
        "policy_coefficients": pd.concat([coefficient_table(fit, m) for m, fit in fits.items()], ignore_index=True),
        "proxy_gap": proxy_gap(data, oof_mean),
        "fairness_audit": fairness_audit(data, reference.to_numpy()),
        **history_tables(data), **simulation_audit(promotions, truth),
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, float_format="%.17g")
    return tables
