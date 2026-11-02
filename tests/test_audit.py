import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from Python.analysis import ROOT, load_data
from Python.audit import (
    MIN_DENOMINATOR, SPECS, AuditPrep, fairness_audit, katz_ratio, laplace_fit, nadeau_bengio,
    wilson,
)
from scripts.prepare_data import boat_side

OUTPUT = ROOT / "outputs" / "python"


def test_nadeau_bengio_inflates_naive_standard_error():
    d = np.array([.01, .03, -.02, .04, .02, .00, .01, .05, -.01, .03])
    mean, sd, se, lower, upper = nadeau_bengio(d, 1 / 4)
    assert np.isclose(se, np.sqrt((1 / 10 + 1 / 4) * d.var(ddof=1)))
    assert se > sd / np.sqrt(len(d))
    q = student_t.ppf(.975, 9)
    assert np.isclose(lower, mean - q * se) and np.isclose(upper, mean + q * se)


def test_wilson_and_katz_intervals():
    lower, upper = wilson(0, 10)
    assert lower == 0 and 0 < upper < .35
    ratio, ir_lower, ir_upper = katz_ratio(20, 100, 40, 100)
    assert np.isclose(ratio, .5) and ir_lower < .5 < ir_upper
    assert np.isnan(wilson(0, 0)[0])


def test_boat_side_mapping():
    assert boat_side("7") == "Starboard" and boat_side("C") == "Starboard"
    assert boat_side("6") == "Port" and boat_side("D") == "Port"
    assert boat_side("13 15") == "Starboard"
    assert boat_side("5 7 8") == "Unknown"
    assert boat_side(np.nan) == "None recorded"


def test_audit_preprocessing_uses_training_statistics_only():
    data = load_data()
    train = data.query("subset == 'train'")
    prep = AuditPrep(train)
    original = vars(prep).copy()
    shifted = data.copy()
    shifted.loc[shifted.subset.eq("test"), "Age"] = 1e6
    prep.transform(shifted)
    assert vars(prep).keys() == original.keys()
    assert prep.age_median == train.Age.median()


def test_laplace_fit_is_stationary_and_policy_terms_help():
    data = load_data()
    x = AuditPrep(data).transform(data)
    fits = {m: laplace_fit(x[f], data.Survived) for m, f in SPECS.items()}
    assert all(fit["gradient"] < 1e-7 for fit in fits.values())
    assert fits["Policy + structure"]["loglik"] > fits["Policy only"]["loglik"]
    assert fits["Policy + structure"]["loglik"] > fits["Structure only"]["loglik"]


def test_fairness_suppression_and_chouldechova_identity():
    data = load_data()
    rng = np.random.default_rng(3)
    table = fairness_audit(data, rng.uniform(size=len(data)))
    for name in ["tpr", "fpr", "ppv"]:
        small = table[f"{name}_denominator"] < MIN_DENOMINATOR
        assert table.loc[small, name].isna().all()
        assert table.loc[~small, name].notna().all()
    full = table.dropna(subset=["tpr", "fpr", "ppv"]).query("ppv > 0 and ppv < 1")
    assert len(full) > 0
    p = full.base_rate
    implied = p / (1 - p) * (1 - full.ppv) / full.ppv * full.tpr
    np.testing.assert_allclose(full.fpr, implied, rtol=1e-12)


def test_committed_audit_results_tell_the_documented_story():
    comparison = pd.read_csv(OUTPUT / "model_comparison.csv")
    auc = comparison.query("metric == 'roc_auc'").set_index("model")
    assert auc.loc["Policy only", "upper"] < 0 and auc.loc["Structure only", "upper"] < 0
    assert auc.loc["Sex-blind", "upper"] < 0
    recovery = pd.read_csv(OUTPUT / "sex_recovery.csv").set_index("features")
    assert recovery.loc["With title", "mean_auc"] > .99 > recovery.loc["Without title", "mean_auc"]
    gap = pd.read_csv(OUTPUT / "proxy_gap.csv").set_index("source").share_of_observed_gap
    assert gap["Sex-blind"] < .5 < .9 < gap["Sex-blind + title"]
    folds = pd.read_csv(ROOT / "data" / "repeated_cv.csv")
    groups = load_data().set_index("PassengerId").ticket_group
    folds["ticket_group"] = groups.loc[folds.PassengerId].to_numpy()
    assert folds.groupby(["repeat", "ticket_group"]).fold.nunique().max() == 1
    assert folds.groupby("repeat").size().eq(1309).all()


def test_simulation_recovers_injected_penalty_and_proxy():
    truth = json.loads((ROOT / "data" / "synthetic_truth.json").read_text(encoding="utf-8"))
    coefficients = pd.read_csv(OUTPUT / "sim_coefficients.csv").set_index(["model", "term"])
    woman = coefficients.loc[("Gender visible", "woman")]
    assert woman.upper < 0
    assert woman.lower < truth["injected_log_odds"]["woman"] < woman.upper
    assert truth["injected_log_odds"]["part_time"] == 0
    assert coefficients.loc[("Gender-blind", "part_time"), "upper"] < 0
    assert pd.read_csv(OUTPUT / "sim_recovery.csv").test_auc.iloc[0] > .55


def test_no_names_tickets_or_free_text_are_published():
    banned = {"name", "ticket", "cabin", "boat", "body", "home.dest", "home_dest"}
    paths = list((ROOT / "data").glob("*.csv")) + list(OUTPUT.glob("*.csv"))
    for path in paths:
        columns = {c.lower() for c in pd.read_csv(path, nrows=0).columns}
        assert not columns & banned, path.name
