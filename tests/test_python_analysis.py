import numpy as np
import pandas as pd
import pytest
from scipy.optimize._numdiff import approx_derivative

from Python.analysis import (
    FEATURES, MODELS, ROOT, BayesianLogistic, Preprocessor, association_rules, load_data,
    posterior_objective, reliability, score, transactions,
)


def test_apriori_counts_baselines_filters_and_singleton_outcome():
    baskets = [
        frozenset(["Sex=female", "Class=1", "Outcome=Survived"]),
        frozenset(["Sex=female", "Class=1", "Outcome=Survived"]),
        frozenset(["Sex=male", "Class=1", "Outcome=Did not survive"]),
        frozenset(["Sex=male", "Class=3", "Outcome=Did not survive"]),
    ]
    rules = association_rules(baskets, .25, .6, 1, 1)
    rule = rules.loc[rules.antecedent.eq("Class=1 & Sex=female")].iloc[0]
    assert (rule["count"], rule.antecedent_count, rule.consequent_count, rule.N) == (2, 2, 2, 4)
    assert (rule.support, rule.confidence, rule.baseline, rule.lift) == (.5, 1, .5, 2)
    assert not rules.antecedent.str.contains("Outcome=").any()
    assert not rules.consequent.str.contains(" & ").any()
    assert association_rules(baskets, .25, .6, 3, 1).empty


def test_true_normal_prior_gradient_hessian_and_stationary_laplace_mode():
    rng = np.random.default_rng(17)
    x = np.column_stack([np.ones(40), rng.normal(size=(40, 2))])
    y = np.tile([0., 1.], 20)
    theta = np.array([.7, -.4, .3])
    prior = np.array([5., 2.5, 2.5])
    value, gradient, hessian = posterior_objective(theta, x, y, prior)
    numeric_gradient = approx_derivative(
        lambda t: np.array([posterior_objective(t, x, y, prior)[0]]), theta).ravel()
    numeric_hessian = approx_derivative(lambda t: posterior_objective(t, x, y, prior)[1], theta)
    np.testing.assert_allclose(gradient, numeric_gradient, atol=1e-7)
    np.testing.assert_allclose(hessian, numeric_hessian, atol=1e-7)
    zero_value, _, _ = posterior_objective(theta, np.zeros_like(x), y, prior)
    assert np.isclose(zero_value, len(y) * np.log(2) + .5 * np.sum((theta / prior) ** 2))
    fitted = BayesianLogistic().fit(pd.DataFrame(x[:, 1:]), y)
    assert fitted.gradient_norm < 1e-6
    assert np.linalg.eigvalsh(fitted.covariance).min() > 0


def test_preprocessing_learns_training_only_and_relearns_inside_folds():
    data = load_data()
    train, test = data.query("subset == 'train'"), data.query("subset == 'test'").copy()
    prep = Preprocessor.fit(train)
    original = prep.as_dict()
    expected = prep.transform(train)
    test.loc[:, "Age"] = 99999
    test.loc[:, "FarePerPerson"] = 1e12
    prep.transform(test)
    assert prep.as_dict() == original
    np.testing.assert_allclose(expected.iloc[:, :3].mean(), 0, atol=1e-14)
    np.testing.assert_allclose(expected.iloc[:, :3].std(ddof=1), 1, atol=1e-14)
    assert expected.columns.tolist() == FEATURES
    inner = train.loc[train.cv_fold.ne(0)]
    inner_prep = Preprocessor.fit(inner)
    assert inner_prep.age_median == inner.Age.median()
    assert inner_prep.means["log_fare_z"] == np.log1p(inner.FarePerPerson.fillna(inner.FarePerPerson.median())).mean()
    assert inner_prep.as_dict() != prep.as_dict()


def test_mode_ties_missingness_zero_variance_and_rules_unknowns():
    frame = pd.DataFrame({
        "Age": [np.nan, 20, 20, 20], "FarePerPerson": [np.nan, 10, 10, 10],
        "Embarked": ["Q", "C", "Q", "C"], "Sex": ["female"] * 4,
        "Pclass": [1] * 4, "SibSp": [0] * 4, "Parch": [0] * 4, "Survived": [1] * 4,
    })
    prep = Preprocessor.fit(frame)
    assert prep.embarked_mode == "C"
    assert prep.scales == dict.fromkeys(FEATURES[:3], 1.)
    assert prep.transform(frame).age_missing.tolist() == [1, 0, 0, 0]
    basket = transactions(frame)[0]
    assert {"Age=Unknown", "Fare=Unknown", "Family=Alone", "Outcome=Survived"} <= basket


def test_committed_source_group_separation_and_coverage():
    data = load_data()
    assert len(data) == 1309
    assert data.PassengerId.nunique() == 1309
    assert data.query("subset == 'train'").shape[0] == 1048
    assert data.InKaggleTrain.sum() == 891
    assert data.groupby("ticket_group").subset.nunique().max() == 1
    assert data.query("subset == 'train'").groupby("ticket_group").cv_fold.nunique().max() == 1


def test_event_direction_threshold_and_calibration_counts():
    y = np.array([0, 0, 1, 1])
    probability = np.array([.1, .2, .5, .9])
    correct, reverse = score(y, probability), score(y, 1 - probability)
    assert correct["roc_auc"] == 1 and reverse["roc_auc"] == 0
    assert correct["balanced_accuracy"] == 1
    assert correct["brier"] < reverse["brier"]
    calibration = reliability(y, probability)
    # right-closed bins, as in R: 0.2 belongs to [0,0.2] and 0.6 to (0.4,0.6]
    assert calibration["n"].tolist() == [2, 0, 1, 0, 1]
    assert reliability([1], [.6])["n"].tolist() == [0, 0, 1, 0, 0]
    assert calibration["n"].sum() == 4


def test_export_contract_matches_source_design_and_event_metrics():
    """The committed public exhibits must remain auditable from their source."""
    data = load_data()
    train, test = data.query("subset == 'train'"), data.query("subset == 'test'")
    output = ROOT / "outputs" / "python"
    predictions = pd.read_csv(output / "predictions.csv")
    assert predictions.columns.tolist() == ["PassengerId", "Survived", "model", "probability"]
    assert len(predictions) == 3 * len(test)
    assert not predictions.duplicated(["PassengerId", "model"]).any()
    metrics = pd.read_csv(output / "metrics.csv")
    assert metrics.columns.tolist() == [
        "phase", "fold", "model", "roc_auc", "brier", "log_loss", "balanced_accuracy", "n",
    ]
    assert len(metrics.query("phase == 'cv'")) == 15
    prep = Preprocessor.fit(train)
    for subset, frame in [("train", train), ("test", test)]:
        design = pd.read_csv(output / f"design_{subset}.csv")
        assert design.columns.tolist() == ["PassengerId"] + FEATURES
        assert design.PassengerId.tolist() == frame.PassengerId.tolist()
        np.testing.assert_allclose(design[FEATURES], prep.transform(frame), atol=1e-14)
    for model in MODELS:
        group = predictions.loc[predictions.model.eq(model)]
        assert group.PassengerId.tolist() == test.PassengerId.tolist()
        np.testing.assert_array_equal(group.Survived, test.Survived)
        calculated = score(group.Survived, group.probability)
        exported = metrics.loc[metrics.phase.eq("test") & metrics.model.eq(model)].iloc[0]
        for metric in calculated:
            assert np.isclose(exported[metric], calculated[metric], atol=1e-14)
    pd.testing.assert_frame_equal(
        pd.read_csv(output / "rules.csv"), association_rules(transactions(train)),
        check_dtype=False, atol=1e-14, rtol=1e-14,
    )


@pytest.mark.parametrize("field,value", [
    ("count", .5), ("antecedent_count", 0), ("confidence", .1),
    ("baseline", .9), ("lift", 1),
])
def test_story_rejects_rule_rates_or_counts_that_disagree(monkeypatch, field, value):
    import json
    from scripts import build_story

    original = build_story.read_table

    def altered_table(path, **kwargs):
        rows = original(path, **kwargs)
        if path.name == "rules.csv":
            rows[0][field] = value
        return rows

    monkeypatch.setattr(build_story, "read_table", altered_table)
    summary = json.loads((ROOT / "outputs" / "python" / "summary.json").read_text())
    with pytest.raises(build_story.ContractError, match="rule|Rule"):
        build_story.load_edition(ROOT / "outputs" / "python", summary)
