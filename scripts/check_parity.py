"""Check shared evidence, not equality of two different forest implementations."""

import csv
import math
from pathlib import Path

from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]


def table(edition, name):
    with (ROOT / "outputs" / edition / f"{name}.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def compare(name, keys, numbers, tolerance=1e-8):
    def indexed(edition):
        rows = table(edition, name)
        result = {tuple(row[k] for k in keys): row for row in rows}
        if len(result) != len(rows):
            raise AssertionError(f"Duplicate keys in {edition}/{name}")
        return result

    r_rows, py_rows = indexed("r"), indexed("python")
    if r_rows.keys() != py_rows.keys():
        raise AssertionError(f"R/Python keys differ for {name}")
    maximum = 0.0
    for key, r_row in r_rows.items():
        for field in numbers:
            a, b = float(r_row[field]), float(py_rows[key][field])
            if not math.isfinite(a) or not math.isfinite(b):
                raise AssertionError(f"Nonfinite {name}/{key}/{field}")
            maximum = max(maximum, abs(a - b))
            if not math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance):
                raise AssertionError(f"R/Python mismatch {name}/{key}/{field}: {a} vs {b}")
    print(f"{name}: {len(r_rows)} matching rows; maximum absolute difference {maximum:.3g}")


def check_evaluation():
    with (ROOT / "data" / "split.csv").open(newline="") as handle:
        split = list(csv.DictReader(handle))
    with (ROOT / "data" / "passengers.csv").open(newline="") as handle:
        outcomes = {row["PassengerId"]: int(row["Survived"]) for row in csv.DictReader(handle)}
    test_ids = {row["PassengerId"] for row in split if row["subset"] == "test"}
    for edition in ("r", "python"):
        predictions = table(edition, "predictions")
        metrics = table(edition, "metrics")
        for model in ("Prevalence baseline", "Random forest", "Bayesian logistic"):
            rows = [row for row in predictions if row["model"] == model]
            if len(rows) != len(test_ids) or {r["PassengerId"] for r in rows} != test_ids:
                raise AssertionError(f"Held-out prediction IDs differ: {edition}/{model}")
            actual = [int(row["Survived"]) for row in rows]
            if actual != [outcomes[row["PassengerId"]] for row in rows]:
                raise AssertionError(f"Outcome labels differ: {edition}/{model}")
            probability = [float(row["probability"]) for row in rows]
            if not all(math.isfinite(p) and 0 <= p <= 1 for p in probability):
                raise AssertionError(f"Invalid probability: {edition}/{model}")
            expected = {
                "roc_auc": roc_auc_score(actual, probability),
                "brier": brier_score_loss(actual, probability),
                "log_loss": log_loss(actual, probability, labels=[0, 1]),
                "balanced_accuracy": balanced_accuracy_score(actual, [p >= .5 for p in probability]),
            }
            recorded = [r for r in metrics if r["phase"] == "test" and r["model"] == model]
            if len(recorded) != 1:
                raise AssertionError(f"Missing/duplicate test metrics: {edition}/{model}")
            for field, value in expected.items():
                if not math.isclose(value, float(recorded[0][field]), abs_tol=1e-8, rel_tol=1e-8):
                    raise AssertionError(f"Metric/prediction mismatch: {edition}/{model}/{field}")
    print("Both editions: held-out IDs/outcomes and all exported test metrics verified.")


def number(value):
    try:
        return float(value)
    except ValueError:
        return None


def normalized_key(row, keys):
    return tuple(
        number(row[k]) if number(row[k]) is not None else row[k] for k in keys
    )


def compare_all(name, keys, tolerance=1e-9, exact=(), skip_forest=True, auc_pairs=None,
                row_tolerance=None):
    """Compare every column: identical headers, strings exactly, counts exactly,
    other numbers within `tolerance` (relative or absolute); empty must match empty.

    `auc_pairs(key)` returns n_positive * n_negative for a row. ROC AUC is discontinuous at
    exact ties: identical design rows have identical mathematical probabilities, but numpy's
    blocked matmul can place them one ulp apart, so sklearn counts a tied pair as ordered.
    An AUC difference is then accepted only if it is a whole number of half-pair weights
    (0.5 / pairs) - i.e. explained solely by flipped exact ties - and every flip is reported.
    """
    def load(edition):
        with (ROOT / "outputs" / edition / f"{name}.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames
            rows = [
                row for row in reader
                if not (skip_forest and any(row.get(c) == "Random forest" for c in ("model", "source")))
            ]
        result = {normalized_key(row, keys): row for row in rows}
        if len(result) != len(rows):
            raise AssertionError(f"Duplicate keys in {edition}/{name}")
        return header, result

    r_header, r_rows = load("r")
    py_header, py_rows = load("python")
    if r_header != py_header:
        raise AssertionError(f"R/Python columns differ for {name}: {r_header} vs {py_header}")
    if r_rows.keys() != py_rows.keys():
        missing = sorted(map(str, set(py_rows) ^ set(r_rows)))[:5]
        raise AssertionError(f"R/Python keys differ for {name}: e.g. {missing}")
    maximum = 0.0
    flips = []
    for key, r_row in r_rows.items():
        for field in py_header:
            a, b = r_row[field], py_rows[key][field]
            if a == "" or b == "":
                if a != b:
                    raise AssertionError(f"R/Python missingness differs {name}/{key}/{field}: {a!r} vs {b!r}")
                continue
            x, y = number(a), number(b)
            if x is None or y is None:
                if a != b:
                    raise AssertionError(f"R/Python string mismatch {name}/{key}/{field}: {a!r} vs {b!r}")
                continue
            if not math.isfinite(x) or not math.isfinite(y):
                raise AssertionError(f"Nonfinite {name}/{key}/{field}")
            limit = 0.0 if field in exact else (row_tolerance or (lambda k: tolerance))(key)
            if math.isclose(x, y, rel_tol=limit, abs_tol=limit):
                maximum = max(maximum, abs(x - y))
                continue
            if auc_pairs is not None and field == "roc_auc":
                half_pairs = 2 * abs(x - y) * auc_pairs(key)
                if 1 <= round(half_pairs) <= MAX_TIE_FLIPS and abs(half_pairs - round(half_pairs)) < 1e-6:
                    flips.append(f"{key}: {round(half_pairs)} tied pair(s), diff {abs(x - y):.3g}")
                    continue
            raise AssertionError(f"R/Python mismatch {name}/{key}/{field}: {x!r} vs {y!r}")
    print(f"{name}: {len(r_rows)} matching rows x {len(py_header)} columns; "
          f"maximum absolute difference {maximum:.3g} (tolerance {tolerance:g})")
    for flip in flips:
        print(f"  roc_auc exact-tie flip (numpy ulp ordering) {flip}")


# Python's main posterior fit stops at trust-exact gtol 1e-8 (accepting gradients up to 1e-6),
# so its plug-in probabilities agree with R's polished mode only to about 1e-7.
MAIN_BAYES = 2e-5
# Audit Laplace fits are both polished to max|gradient| <= 1e-10: much tighter agreement.
LAPLACE = 2e-6
COUNT = 1e-12
MAX_TIE_FLIPS = 20
# model_comparison roc_auc rows average the 100 fold AUCs, a few of which differ by whole
# tied-pair weights (~3e-5 each, reported under cv_scores); their mean/sd/se therefore
# differ by up to (sum of flips) / 100, about 4e-6 here. All other rows use LAPLACE.
AUC_AGGREGATE = 1e-5


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_pairs():
    """n_survived * n_died for every exported CV fold and the held-out test subset."""
    pairs = {}
    for name, fold_field in [("cv_predictions", "fold"), ("predictions", None)]:
        groups = {}
        for row in table("python", name):
            if row["model"] != "Bayesian logistic":
                continue
            fold = float(row[fold_field]) if fold_field else -1.0
            groups.setdefault(fold, []).append(int(row["Survived"]))
        for fold, y in groups.items():
            pairs[fold] = sum(y) * (len(y) - sum(y))
    return lambda key: pairs[key[1]]


def audit_pairs():
    """Positive x negative counts per repeated-CV fold, for survival and recorded-sex tasks."""
    people = {row["PassengerId"]: row for row in csv_rows(ROOT / "data" / "passengers.csv")}
    counts = {}
    for row in csv_rows(ROOT / "data" / "repeated_cv.csv"):
        person = people[row["PassengerId"]]
        entry = counts.setdefault((float(row["repeat"]), float(row["fold"])), [0, 0, 0])
        entry[0] += 1
        entry[1] += int(person["Survived"])
        entry[2] += person["Sex"] == "male"
    def pairs(key):
        n, survived, male = counts[(key[1], key[2])]
        positive = survived if key[0] == "Predict survival" else male
        return positive * (n - positive)
    return pairs


def check_tie_collapsed_auc():
    """Collapsing one-ulp numerical ties makes both editions' CV AUCs equal R's exact-tie AUC."""
    recorded = {(float(r["fold"]), r["model"]): float(r["roc_auc"])
                for r in table("r", "metrics") if r["phase"] == "cv"}
    maximum = 0.0
    for edition in ("python", "r"):
        groups = {}
        for row in table(edition, "cv_predictions"):
            if row["model"] != "Random forest":
                groups.setdefault((float(row["fold"]), row["model"]), []).append(row)
        for key, rows in groups.items():
            collapsed = roc_auc_score([int(r["Survived"]) for r in rows],
                                      [float(f"{float(r['probability']):.12g}") for r in rows])
            maximum = max(maximum, abs(collapsed - recorded[key]))
            if not math.isclose(collapsed, recorded[key], rel_tol=1e-12, abs_tol=1e-12):
                raise AssertionError(f"Tie-collapsed AUC differs: {edition}/{key}")
    print(f"metrics: tie-collapsed CV AUC from both editions' cv_predictions equals R's "
          f"(pROC, exact ties); maximum absolute difference {maximum:.3g}")


def main():
    check_evaluation()
    features = [
        "age_z", "log_fare_z", "family_size_z", "age_missing", "sex_male",
        "class_2", "class_3", "embarked_Q", "embarked_S",
    ]
    compare("design_train", ["PassengerId"], features)
    compare("design_test", ["PassengerId"], features)
    compare("coefficients", ["term"],
            ["estimate", "sd", "lower", "upper", "odds_ratio", "or_lower", "or_upper"],
            tolerance=2e-5)
    compare("rules", ["antecedent", "consequent"],
            ["support", "confidence", "lift", "count", "antecedent_count",
             "consequent_count", "baseline", "N"])
    compare("bayesian_rates", ["sex", "pclass"],
            ["n", "survived", "observed_rate", "posterior_mean", "lower", "upper"])
    compare_all("missingness", ["field"], COUNT, exact={"n_missing"})
    compare_all("calibration", ["model", "bin"], MAIN_BAYES, exact={"n"})
    compare_all("predictions", ["PassengerId", "model"], MAIN_BAYES, exact={"Survived"})
    compare_all("cv_predictions", ["PassengerId", "model"], MAIN_BAYES, exact={"Survived", "fold"})
    compare_all("metrics", ["phase", "fold", "model"], MAIN_BAYES, exact={"n"},
                auc_pairs=metric_pairs())
    check_tie_collapsed_auc()

    compare_all("cv_scores", ["task", "repeat", "fold", "model"], LAPLACE, exact={"n"},
                auc_pairs=audit_pairs())
    compare_all("oof_mean", ["PassengerId", "model"], LAPLACE)
    compare_all("model_comparison", ["model", "metric"], LAPLACE, exact={"folds"},
                row_tolerance=lambda key: AUC_AGGREGATE if key[1] == "roc_auc" else LAPLACE)
    compare_all("sex_recovery", ["features"], LAPLACE, exact={"folds"})
    compare_all("policy_fit", ["model"], LAPLACE, exact={"terms"})
    compare_all("policy_coefficients", ["model", "term"], LAPLACE)
    compare_all("proxy_gap", ["source"], LAPLACE)
    compare_all("fairness_audit", ["grouping", "group"], LAPLACE,
                exact={"n", "tpr_denominator", "fpr_denominator", "ppv_denominator"})
    compare_all("policy_groups", ["group", "pclass"], COUNT, exact={"n", "survived"})
    compare_all("child_cutoffs", ["cutoff", "pclass"], COUNT, exact={"n", "survived"})
    compare_all("boat_side", ["side"], COUNT, exact={"n", "adult_men"})
    compare_all("missing_age", ["pclass"], COUNT, exact={"n", "missing_age"})
    compare_all("sim_coefficients", ["model", "term"], LAPLACE)
    compare_all("sim_gap", ["source"], LAPLACE)
    compare_all("sim_recovery", ["target"], LAPLACE, exact={"test_n"})
    compare_all("sim_impact", ["label", "grouping", "group"], COUNT, exact={"n", "promoted"})
    print("Forest probability parity is intentionally not asserted: the implementations differ.")


if __name__ == "__main__":
    main()
