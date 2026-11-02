"""Bundle committed result tables into the offline story (docs/index.html).

Every number shown on the page comes from a CSV/JSON loaded and validated here.
The audit chapters depend only on outputs/python. The appendix also offers the R
edition when outputs/r matches the shared split; otherwise it is omitted.
"""

import base64
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "outputs" / "python"
MODELS = {"Prevalence baseline", "Random forest", "Bayesian logistic"}
AUDIT_MODELS = ["Prevalence baseline", "Policy only", "Structure only", "Policy + structure",
                "Sex-blind", "Sex-blind + title", "Random forest"]
REFERENCE = "Policy + structure"
MIN_DENOMINATOR = 20


class ContractError(ValueError):
    pass


def check(condition, message):
    if not condition:
        raise ContractError(message)


def read_table(path, text=(), columns=None, rows=None):
    """Read a CSV, converting numbers; empty cells become None (suppressed values)."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        raw = list(reader)
    if columns is not None:
        missing = set(columns) - set(fields)
        check(not missing, f"{path.name}: missing columns {sorted(missing)}")
    if rows is not None:
        check(len(raw) == rows, f"{path.name}: expected {rows} rows, found {len(raw)}")
    result = []
    for row in raw:
        converted = {}
        for field, value in row.items():
            if field in text:
                converted[field] = value
            elif value == "":
                converted[field] = None
            else:
                try:
                    number = float(value)
                except ValueError:
                    converted[field] = value
                else:
                    check(math.isfinite(number), f"Nonfinite value in {path}: {field}")
                    converted[field] = number
        result.append(converted)
    return result


def check_interval(rows, name, est="rate", lo="lower", hi="upper", tol=1e-9):
    for r in rows:
        if r[est] is None:
            continue
        check(r[lo] - tol <= r[est] <= r[hi] + tol, f"{name}: estimate outside its interval")


def check_probabilities(rows, name, *fields):
    for r in rows:
        for f in fields:
            if r[f] is not None:
                check(-1e-9 <= r[f] <= 1 + 1e-9, f"{name}: {f} is not a probability")


def load_edition(folder, summary):
    """Appendix tables (original lenses) for one language."""
    test_n, train_n = int(summary["test_n"]), int(summary["train_n"])
    tables = {
        "metrics": read_table(folder / "metrics.csv", text=("phase", "model"), rows=18),
        "predictions": read_table(folder / "predictions.csv", text=("model",), rows=test_n * 3),
        "coefficients": read_table(folder / "coefficients.csv", text=("term",)),
        "rules": read_table(folder / "rules.csv", text=("antecedent", "consequent"),
                           columns=("antecedent", "consequent", "support", "confidence", "lift",
                                    "count", "antecedent_count", "consequent_count", "baseline", "N")),
        "bayesian_rates": read_table(folder / "bayesian_rates.csv", text=("sex",), rows=6),
        "calibration": read_table(folder / "calibration.csv", text=("model", "bin"), rows=15,
                                  columns=("model", "bin", "n", "predicted", "observed")),
    }
    metrics = tables["metrics"]
    check({r["model"] for r in metrics} == MODELS, f"Unexpected metric models in {folder}")
    for model in MODELS:
        cv = [r for r in metrics if r["model"] == model and r["phase"] == "cv"]
        test = [r for r in metrics if r["model"] == model and r["phase"] == "test"]
        check({r["fold"] for r in cv} == set(range(5)) and len(test) == 1
              and test[0]["n"] == test_n, f"Evaluation schema differs (holdout n={test_n}): {folder}")
    for row in tables["predictions"]:
        check(0 <= row["probability"] <= 1 and row["Survived"] in (0, 1),
              f"Invalid held-out probability or label in {folder}")
    check(len(tables["rules"]) > 0, f"No exported rules in {folder}")
    for row in tables["rules"]:
        check(row["N"] == train_n and "Outcome=" not in row["antecedent"],
              f"Training-only rule contract failed (N={train_n}) in {folder}")
        check(all(row[k] is not None and row[k] == int(row[k])
                  for k in ("count", "antecedent_count", "consequent_count", "N"))
              and 0 < row["count"] <= row["antecedent_count"] <= train_n
              and row["count"] <= row["consequent_count"] <= train_n,
              f"Invalid rule evidence counts in {folder}")
        for field, expected in (
            ("support", row["count"] / train_n),
            ("confidence", row["count"] / row["antecedent_count"]),
            ("baseline", row["consequent_count"] / train_n),
            ("lift", (row["count"] / row["antecedent_count"])
             / (row["consequent_count"] / train_n)),
        ):
            check(row[field] is not None and math.isclose(row[field], expected, rel_tol=1e-8, abs_tol=1e-8),
                  f"Rule {field} does not match evidence counts in {folder}")
    for model in MODELS:
        bins = [r for r in tables["calibration"] if r["model"] == model]
        check(len(bins) == 5 and sum(r["n"] for r in bins) == test_n,
              f"Calibration bins must cover the {test_n}-row holdout: {folder}")
    check_interval(tables["bayesian_rates"], "bayesian_rates", est="posterior_mean")
    return tables


def load_audit(provenance):
    """The decision-audit tables (Python reference outputs only)."""
    t = {}
    t["policy_groups"] = read_table(PY / "policy_groups.csv", text=("group", "pclass"), rows=8,
                                    columns=("group", "pclass", "n", "survived", "rate", "lower", "upper"))
    check({(r["group"], r["pclass"]) for r in t["policy_groups"]}
          == {(g, c) for g in ("Women and children", "Men") for c in ("All", "1", "2", "3")},
          "policy_groups: unexpected groups")
    check(sum(r["n"] for r in t["policy_groups"] if r["pclass"] == "All") == provenance["rows"],
          "policy_groups: overall groups must cover every passenger")

    t["child_cutoffs"] = read_table(PY / "child_cutoffs.csv", text=("pclass",), rows=16,
                                    columns=("cutoff", "pclass", "n", "survived", "rate", "lower", "upper"))
    check({r["cutoff"] for r in t["child_cutoffs"]} == {12, 14, 16, 18}, "child_cutoffs: cutoffs")

    t["boat_side"] = read_table(PY / "boat_side.csv", text=("side",),
                                columns=("side", "n", "adult_men", "adult_men_share", "lower", "upper"))
    check({"Starboard", "Port"} <= {r["side"] for r in t["boat_side"]}, "boat_side: sides")
    check_interval(t["boat_side"], "boat_side", est="adult_men_share")

    t["missing_age"] = read_table(PY / "missing_age.csv", text=("pclass",), rows=4,
                                  columns=("pclass", "n", "missing_age", "missing_share", "lower", "upper"))
    check(next(r for r in t["missing_age"] if r["pclass"] == "All")["n"] == provenance["rows"],
          "missing_age: overall row must cover every passenger")
    check_interval(t["missing_age"], "missing_age", est="missing_share")

    t["policy_fit"] = read_table(PY / "policy_fit.csv", text=("model",), rows=5,
                                 columns=("model", "terms", "loglik", "mcfadden_r2"))

    t["model_comparison"] = read_table(
        PY / "model_comparison.csv", text=("model", "metric", "reference"), rows=21,
        columns=("model", "metric", "reference", "mean_model", "mean_difference", "lower", "upper", "folds"))
    mc = t["model_comparison"]
    check({r["model"] for r in mc} == set(AUDIT_MODELS), "model_comparison: models")
    check({r["metric"] for r in mc} == {"roc_auc", "brier", "log_loss"}, "model_comparison: metrics")
    check(all(r["reference"] == REFERENCE and r["folds"] == 100 for r in mc),
          "model_comparison: 20x5 paired comparison against the reference")
    check_interval(mc, "model_comparison", est="mean_difference")

    t["proxy_gap"] = read_table(PY / "proxy_gap.csv", text=("source",), rows=4,
                                columns=("source", "female_mean", "male_mean", "gap", "share_of_observed_gap"))
    check({r["source"] for r in t["proxy_gap"]}
          == {"Observed outcomes", REFERENCE, "Sex-blind", "Sex-blind + title"}, "proxy_gap: sources")

    t["sex_recovery"] = read_table(PY / "sex_recovery.csv", text=("features",), rows=2,
                                   columns=("features", "mean_auc", "min_auc", "max_auc", "folds"))
    check({r["features"] for r in t["sex_recovery"]} == {"Without title", "With title"}, "sex_recovery")

    fa = read_table(PY / "fairness_audit.csv",
                    text=("model", "grouping", "group", "label_reference", "selection_reference"), rows=11)
    check({r["grouping"] for r in fa} == {"Sex", "Class", "Sex x class"}, "fairness_audit: groupings")
    for metric in ("tpr", "fpr", "ppv"):
        for r in fa:
            suppressed = r[f"{metric}_denominator"] < MIN_DENOMINATOR
            check((r[metric] is None) == suppressed,
                  f"fairness_audit: {metric} must be suppressed exactly when n<{MIN_DENOMINATOR}")
    check_probabilities(fa, "fairness_audit", "base_rate", "mean_predicted", "selection_rate", "tpr", "fpr", "ppv")
    check_interval(fa, "fairness_audit", est="impact_ratio", lo="ir_lower", hi="ir_upper")
    check_interval(fa, "fairness_audit", est="selection_rate", lo="selection_lower", hi="selection_upper")
    t["fairness_audit"] = fa

    t["sim_coefficients"] = read_table(PY / "sim_coefficients.csv", text=("model", "term"), rows=11,
                                       columns=("model", "term", "estimate", "lower", "upper", "injected"))
    t["sim_gap"] = read_table(PY / "sim_gap.csv", text=("source",), rows=4,
                              columns=("source", "women_mean", "men_mean", "gap"))
    t["sim_impact"] = read_table(PY / "sim_impact.csv", text=("label", "grouping", "group", "reference"), rows=8,
                                 columns=("label", "grouping", "group", "n", "rate", "impact_ratio", "ir_lower", "ir_upper"))
    t["sim_recovery"] = read_table(PY / "sim_recovery.csv", text=("target", "features"), rows=1,
                                   columns=("target", "features", "test_auc", "test_n"))
    check_interval(t["sim_coefficients"], "sim_coefficients", est="estimate")
    check_interval(t["sim_impact"], "sim_impact", est="impact_ratio", lo="ir_lower", hi="ir_upper")
    return t


def load_synthetic(audit):
    truth = json.loads((ROOT / "data" / "synthetic_truth.json").read_text(encoding="utf-8"))
    check(truth.get("synthetic") is True, "synthetic_truth.json must declare synthetic data")
    for row in audit["sim_coefficients"]:
        if row["injected"] is not None:
            check(abs(row["injected"] - truth["injected_log_odds"][row["term"]]) < 1e-12,
                  "sim_coefficients: injected values differ from synthetic_truth.json")
    rows = read_table(ROOT / "data" / "synthetic_promotions.csv", text=("gender", "site", "subset"),
                      rows=truth["n"], columns=("gender", "part_time", "subset"))
    part_time = {}
    for gender in ("Woman", "Man"):
        group = [r for r in rows if r["gender"] == gender]
        part_time[gender] = {"n": len(group), "part_time": int(sum(r["part_time"] for r in group))}
    test_n = sum(r["subset"] == "test" for r in rows)
    check(test_n == audit["sim_recovery"][0]["test_n"], "Synthetic test subset differs from sim_recovery")
    return {"n": truth["n"], "injected": truth["injected_log_odds"], "warning": truth["warning"],
            "part_time": part_time, "test_n": test_n}


def font_face(file, style):
    data = base64.b64encode((ROOT / "web" / "assets" / file).read_bytes()).decode("ascii")
    return (f'@font-face{{font-family:"Instrument Serif";font-style:{style};font-weight:400;'
            f'font-display:swap;src:url(data:font/woff2;base64,{data}) format("woff2");}}')


def build_payload():
    provenance = json.loads((ROOT / "data" / "provenance.json").read_text(encoding="utf-8"))
    summary = json.loads((PY / "summary.json").read_text(encoding="utf-8"))
    check(summary["n"] == provenance["rows"] and summary["test_n"] + summary["train_n"] == summary["n"],
          "summary.json does not match provenance")
    audit = load_audit(provenance)
    languages = {"Python": load_edition(PY, summary)}
    notes = {}
    try:
        languages["R"] = load_edition(ROOT / "outputs" / "r", summary)
    except (ContractError, OSError, KeyError) as error:
        notes["R"] = str(error)
        print(f"R edition omitted from the appendix: {error}")
    return {
        "provenance": {k: provenance[k] for k in
                       ("required_credit", "rows", "survived", "coverage", "source_name", "source_page")},
        "summary": summary,
        "audit": audit,
        "synthetic": load_synthetic(audit),
        "languages": languages,
        "language_notes": notes,
    }


def main():
    payload = build_payload()
    encoded = json.dumps(payload, ensure_ascii=True, allow_nan=False)
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    fonts = font_face("instrument-serif-latin-400-normal.woff2", "normal") + font_face(
        "instrument-serif-latin-400-italic.woff2", "italic")
    template = (ROOT / "web" / "story.html").read_text(encoding="utf-8")
    for token in ("/*__FONTS__*/", "__DATA__"):
        check(template.count(token) == 1, f"Template must contain {token} exactly once")
    html = template.replace("/*__FONTS__*/", fonts).replace("__DATA__", encoded)
    output = ROOT / "docs"
    output.mkdir(exist_ok=True)
    (output / "index.html").write_text(html, encoding="utf-8")
    print(f"Built {output / 'index.html'} ({len(html):,} characters)")


if __name__ == "__main__":
    main()
