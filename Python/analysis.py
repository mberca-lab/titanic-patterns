"""Shared transformations, historical summaries, and fixed modeling protocol."""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import beta, norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score,
)

SEED = 20261014
ROOT = Path(__file__).resolve().parents[1]
CONTINUOUS = ["age_z", "log_fare_z", "family_size_z"]
FEATURES = CONTINUOUS + [
    "age_missing", "sex_male", "class_2", "class_3", "embarked_Q", "embarked_S",
]
MODELS = ["Prevalence baseline", "Random forest", "Bayesian logistic"]
RULE_COLUMNS = [
    "antecedent", "consequent", "support", "confidence", "lift", "count",
    "antecedent_count", "consequent_count", "baseline", "N",
]


def load_data(root=ROOT):
    """Read the committed public extract and split without changing row order."""
    data = pd.read_csv(Path(root) / "data" / "passengers.csv")
    split = pd.read_csv(Path(root) / "data" / "split.csv")
    if data.PassengerId.duplicated().any() or split.PassengerId.duplicated().any():
        raise ValueError("Passenger IDs must be unique.")
    if set(data.PassengerId) != set(split.PassengerId):
        raise ValueError("Data/split IDs differ.")
    joined = data.merge(split, on="PassengerId", how="left", validate="one_to_one", sort=False)
    if set(joined.subset) != {"train", "test"}:
        raise ValueError("Expected train and test assignments.")
    if not joined.Survived.isin([0, 1]).all():
        raise ValueError("Survival event must be coded 0/1.")
    if joined.groupby("ticket_group").subset.nunique().max() != 1:
        raise ValueError("Ticket group crosses holdout boundary.")
    train = joined.loc[joined.subset.eq("train")]
    if set(train.cv_fold) != set(range(5)):
        raise ValueError("Expected five committed training folds.")
    if train.groupby("ticket_group").cv_fold.nunique().max() != 1:
        raise ValueError("Ticket group crosses validation folds.")
    return joined


@dataclass
class Preprocessor:
    age_median: float
    fare_median: float
    embarked_mode: str
    means: dict
    scales: dict

    @classmethod
    def fit(cls, data):
        modes = data.Embarked.dropna().mode().sort_values()
        if modes.empty or data.Age.notna().sum() == 0 or data.FarePerPerson.notna().sum() == 0:
            raise ValueError("Training data cannot supply imputation statistics.")
        result = cls(float(data.Age.median()), float(data.FarePerPerson.median()), str(modes.iloc[0]), {}, {})
        raw = result.continuous(data)
        result.means = {c: float(raw[c].mean()) for c in CONTINUOUS}
        result.scales = {
            c: float(raw[c].std(ddof=1)) if raw[c].std(ddof=1) > 0 else 1.0
            for c in CONTINUOUS
        }
        return result

    def continuous(self, data):
        # Fare is recorded per ticket; FarePerPerson divides it by the ticket's group size.
        return pd.DataFrame({
            "age_z": data.Age.fillna(self.age_median),
            "log_fare_z": np.log1p(data.FarePerPerson.fillna(self.fare_median)),
            "family_size_z": data.SibSp + data.Parch + 1,
        }, index=data.index)

    def transform(self, data):
        x = self.continuous(data)
        for col in CONTINUOUS:
            x[col] = (x[col] - self.means[col]) / self.scales[col]
        x["age_missing"] = data.Age.isna().astype(float)
        x["sex_male"] = data.Sex.eq("male").astype(float)
        x["class_2"] = data.Pclass.eq(2).astype(float)
        x["class_3"] = data.Pclass.eq(3).astype(float)
        embarked = data.Embarked.fillna(self.embarked_mode)
        x["embarked_Q"] = embarked.eq("Q").astype(float)
        x["embarked_S"] = embarked.eq("S").astype(float)
        return x[FEATURES].astype(float)

    def as_dict(self):
        return {
            "age_median": self.age_median, "fare_median": self.fare_median,
            "embarked_mode": self.embarked_mode, "means": self.means,
            "scales": self.scales, "features": FEATURES,
        }


def posterior_objective(theta, x, y, prior_sd):
    """Negative log posterior (up to a constant), gradient, analytic Hessian."""
    z = x @ theta
    p = expit(z)
    precision = 1 / np.square(prior_sd)
    value = np.sum(np.logaddexp(0, z) - y * z) + 0.5 * np.dot(theta * precision, theta)
    gradient = x.T @ (p - y) + precision * theta
    hessian = x.T @ ((p * (1 - p))[:, None] * x) + np.diag(precision)
    return value, gradient, hessian


class BayesianLogistic:
    """Normal-prior posterior mode and local Gaussian (Laplace) approximation."""

    def fit(self, x, y):
        design = np.column_stack([np.ones(len(x)), np.asarray(x)])
        y = np.asarray(y, dtype=float)
        self.prior_sd = np.array([5.] + [2.5] * x.shape[1])
        objective = lambda theta: posterior_objective(theta, design, y, self.prior_sd)
        result = minimize(
            lambda theta: objective(theta)[0],
            np.zeros(design.shape[1]),
            jac=lambda theta: objective(theta)[1],
            hess=lambda theta: objective(theta)[2],
            method="trust-exact", options={"gtol": 1e-8, "maxiter": 300},
        )
        value, gradient, hessian = objective(result.x)
        self.gradient_norm = float(np.max(np.abs(gradient)))
        # Some SciPy versions report radius stagnation at an already stationary mode.
        # Accept only the explicitly checked tight stationarity and positive curvature.
        if self.gradient_norm > 1e-6 or not np.isfinite(value):
            raise RuntimeError(f"Posterior optimization failed: {result.message}; gradient={self.gradient_norm}")
        np.linalg.cholesky(hessian)
        self.theta = result.x
        self.covariance = np.linalg.solve(hessian, np.eye(len(self.theta)))
        self.optimizer_success = bool(result.success)
        self.optimizer_message = str(result.message)
        return self

    def predict_proba(self, x):
        return expit(np.column_stack([np.ones(len(x)), np.asarray(x)]) @ self.theta)

    def coefficients(self):
        sd = np.sqrt(np.diag(self.covariance))
        lower = self.theta - norm.ppf(.975) * sd
        upper = self.theta + norm.ppf(.975) * sd
        return pd.DataFrame({
            "term": ["intercept"] + FEATURES, "estimate": self.theta, "sd": sd,
            "lower": lower, "upper": upper, "odds_ratio": np.exp(self.theta),
            "or_lower": np.exp(lower), "or_upper": np.exp(upper), "prior_sd": self.prior_sd,
        })


def score(y, probability):
    """Event 1 is survival; no automatic ROC direction reversal."""
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "balanced_accuracy": float(balanced_accuracy_score(y, np.asarray(probability) >= .5)),
        "n": len(y),
    }


def fit_predict(train, validation):
    prep = Preprocessor.fit(train)
    x_train, x_valid = prep.transform(train), prep.transform(validation)
    forest = RandomForestClassifier(
        n_estimators=500, max_features=3, min_samples_leaf=5,
        random_state=SEED, n_jobs=2,
    ).fit(x_train, train.Survived)
    bayes = BayesianLogistic().fit(x_train, train.Survived)
    probabilities = {
        MODELS[0]: np.full(len(validation), train.Survived.mean()),
        MODELS[1]: forest.predict_proba(x_valid)[:, list(forest.classes_).index(1)],
        MODELS[2]: bayes.predict_proba(x_valid),
    }
    return probabilities, prep, bayes


def transactions(data):
    """Fixed, ASCII-labeled baskets; unknown ages/fares remain unknown."""
    baskets = []
    for row in data.itertuples():
        age = ("Unknown" if pd.isna(row.Age) else
               "Child (<16)" if row.Age < 16 else "Adult (16-59)" if row.Age < 60 else "Older (60+)")
        family_n = row.SibSp + row.Parch + 1
        family = "Alone" if family_n == 1 else "2-4" if family_n <= 4 else "5+"
        fare = ("Unknown" if pd.isna(row.FarePerPerson) else
                "Low (<=10)" if row.FarePerPerson <= 10 else
                "Middle (10-30)" if row.FarePerPerson <= 30 else "High (>30)")
        baskets.append(frozenset([
            f"Sex={row.Sex}", f"Class={row.Pclass}", f"Age={age}", f"Family={family}",
            f"Fare={fare}", "Outcome=Survived" if row.Survived else "Outcome=Did not survive",
        ]))
    return baskets


def association_rules(baskets, min_support=.04, min_confidence=.60, min_count=20, min_lift=1.0):
    """Apriori with downward-closure pruning, max itemset size three."""
    n = len(baskets)
    if not n:
        return pd.DataFrame(columns=RULE_COLUMNS)
    items = sorted(set().union(*baskets))
    counts = {frozenset([item]): sum(item in basket for basket in baskets) for item in items}
    previous = {key for key, count in counts.items() if count / n >= min_support}
    frequent = set(previous)
    for length in (2, 3):
        pool = sorted(set().union(*previous)) if previous else []
        candidates = [
            frozenset(candidate) for candidate in combinations(pool, length)
            if all(frozenset(subset) in previous for subset in combinations(candidate, length - 1))
        ]
        previous = set()
        for candidate in candidates:
            count = sum(candidate <= basket for basket in baskets)
            counts[candidate] = count
            if count / n >= min_support:
                previous.add(candidate)
        frequent.update(previous)
    rows = []
    for itemset in frequent:
        outcomes = [item for item in itemset if item.startswith("Outcome=")]
        if len(itemset) < 2 or len(outcomes) != 1:
            continue
        consequent = outcomes[0]
        antecedent = itemset - {consequent}
        joint_count, ant_count = counts[itemset], counts[antecedent]
        cons_count = counts[frozenset([consequent])]
        confidence, baseline = joint_count / ant_count, cons_count / n
        lift = confidence / baseline
        if joint_count >= min_count and confidence >= min_confidence and lift >= min_lift:
            rows.append({
                "antecedent": " & ".join(sorted(antecedent)), "consequent": consequent,
                "support": joint_count / n, "confidence": confidence, "lift": lift,
                "count": joint_count, "antecedent_count": ant_count,
                "consequent_count": cons_count, "baseline": baseline, "N": n,
            })
    return pd.DataFrame(rows, columns=RULE_COLUMNS).sort_values(
        ["lift", "support", "antecedent", "consequent"],
        ascending=[False, False, True, True], ignore_index=True,
    )


def descriptive_rates(data):
    rows = []
    for (sex, pclass), group in data.groupby(["Sex", "Pclass"], sort=True):
        n, survived = len(group), int(group.Survived.sum())
        a, b = survived + 1, n - survived + 1
        rows.append({
            "sex": sex, "pclass": pclass, "n": n, "survived": survived,
            "observed_rate": survived / n, "posterior_mean": a / (a + b),
            "lower": beta.ppf(.025, a, b), "upper": beta.ppf(.975, a, b),
            "prior_alpha": 1, "prior_beta": 1,
        })
    return pd.DataFrame(rows)


def missingness(data):
    fields = [c for c in data.columns if c not in ("subset", "cv_fold", "ticket_group")]
    return pd.DataFrame({
        "field": fields, "n_missing": data[fields].isna().sum().to_numpy(),
        "pct_missing": data[fields].isna().mean().to_numpy(),
    })


BIN_LABELS = ["[0,0.2]", "(0.2,0.4]", "(0.4,0.6]", "(0.6,0.8]", "(0.8,1]"]


def reliability(y, probability, n_bins=5):
    """Right-closed equal-width bins (R's cut(include.lowest=TRUE)), including empty bins."""
    y, probability = np.asarray(y), np.asarray(probability)
    breaks = np.arange(n_bins + 1) * (1 / n_bins)  # same floats as R's seq(0, 1, .2)
    bins = np.clip(np.searchsorted(breaks, probability, side="left") - 1, 0, n_bins - 1)
    return pd.DataFrame([
        {"bin": BIN_LABELS[i], "n": int((bins == i).sum()),
         "predicted": float(probability[bins == i].mean()) if (bins == i).any() else np.nan,
         "observed": float(y[bins == i].mean()) if (bins == i).any() else np.nan}
        for i in range(n_bins)
    ])


def run_analysis(root=ROOT):
    root = Path(root)
    output = root / "outputs" / "python"
    output.mkdir(parents=True, exist_ok=True)
    data = load_data(root)
    train, test = data.loc[data.subset.eq("train")], data.loc[data.subset.eq("test")]
    metrics, cv_predictions = [], []
    for fold in range(5):
        fitting, validation = train.loc[train.cv_fold.ne(fold)], train.loc[train.cv_fold.eq(fold)]
        probabilities, _, _ = fit_predict(fitting, validation)
        for model, probability in probabilities.items():
            metrics.append({"phase": "cv", "fold": fold, "model": model, **score(validation.Survived, probability)})
            cv_predictions.extend({
                "PassengerId": int(pid), "Survived": int(y), "model": model,
                "probability": float(p), "fold": fold,
            } for pid, y, p in zip(validation.PassengerId, validation.Survived, probability))
    probabilities, prep, bayes = fit_predict(train, test)
    predictions, calibration = [], []
    for model, probability in probabilities.items():
        metrics.append({"phase": "test", "fold": -1, "model": model, **score(test.Survived, probability)})
        predictions.extend({
            "PassengerId": int(pid), "Survived": int(y), "model": model, "probability": float(p),
        } for pid, y, p in zip(test.PassengerId, test.Survived, probability))
        calibration.append(reliability(test.Survived, probability).assign(model=model)[
            ["model", "bin", "n", "predicted", "observed"]])
    tables = {
        "metrics": pd.DataFrame(metrics), "predictions": pd.DataFrame(predictions),
        "coefficients": bayes.coefficients(),
        "rules": association_rules(transactions(train)),
        "bayesian_rates": descriptive_rates(data), "missingness": missingness(data),
        "calibration": pd.concat(calibration, ignore_index=True),
        "cv_predictions": pd.DataFrame(cv_predictions),
    }
    for subset, frame in [("train", train), ("test", test)]:
        design = prep.transform(frame)
        design.insert(0, "PassengerId", frame.PassengerId)
        tables[f"design_{subset}"] = design
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, float_format="%.17g")
    (output / "preprocessing.json").write_text(json.dumps(prep.as_dict(), indent=2) + "\n", encoding="utf-8")
    summary = {
        "seed": SEED, "n": len(data), "train_n": len(train), "test_n": len(test),
        "survived": int(data.Survived.sum()), "rule_count": len(tables["rules"]),
        "bayesian_max_abs_gradient": bayes.gradient_norm,
        "bayesian_optimizer_success": bayes.optimizer_success,
        "bayesian_optimizer_message": bayes.optimizer_message,
        "prediction_type": "Posterior-mode plug-in; Laplace marginal coefficient intervals",
        "coverage": f"{len(data):,} titanic3 passenger records; crew excluded",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    from .audit import run_audit
    from .figures import overview
    tables.update(run_audit(data, root, output))
    overview(data, tables, output / "overview.png")
    return data, tables, summary
