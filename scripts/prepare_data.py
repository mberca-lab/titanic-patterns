"""Prepare the public titanic3 extract, shared splits, and a known-truth HR simulation.

Names, ticket strings, cabins, boat numbers, body numbers, and home/destination text
are used only in memory (to derive title, ticket groups, per-person fare, and boat side)
and are never written to the repository.
"""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261014
SOURCE = "https://hbiostat.org/data/repo/titanic3.csv"
EXPECTED_SHA256 = "db6df9666818c69a753cd85d743e01502c8518b00579b6aada0d4fc5a66ccb9d"
KAGGLE_REVISION = "53be2fd391100044594de004d559858077a10c43"
KAGGLE_SOURCE = (
    "https://raw.githubusercontent.com/paulhendricks/titanic/"
    f"{KAGGLE_REVISION}/inst/data-raw/train.csv"
)
KAGGLE_SHA256 = "7d118fef8b6ccf7f81111877bc388536f7b1e498a655e3d649d19aaa010e9f6f"
CREDIT = ("Data obtained from http://hbiostat.org/data courtesy of the Vanderbilt "
          "University Department of Biostatistics.")
REPEATS, FOLDS = 20, 5
TITLE_GROUPS = {"Mr": "Mr", "Mrs": "Mrs", "Mme": "Mrs", "Miss": "Miss", "Mlle": "Miss",
                "Ms": "Miss", "Master": "Master"}
STARBOARD = {"1", "3", "5", "7", "9", "11", "13", "15", "A", "C"}
PORT = {"2", "4", "6", "8", "10", "12", "14", "16", "B", "D"}


def fetch(url, expected):
    with urlopen(url, timeout=60) as response:
        raw = response.read()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"Checksum differs from the pinned source: {url}")
    return raw


def title_group(name):
    title = name.split(",", 1)[1].split(".", 1)[0].strip()
    return TITLE_GROUPS.get(title, "Rare")


def boat_side(boat):
    """Starboard (odd, A, C) / port (even, B, D); multi-boat records only if one side."""
    if pd.isna(boat) or not str(boat).strip():
        return "None recorded"
    sides = {"Starboard" if b in STARBOARD else "Port" if b in PORT else "Unknown"
             for b in str(boat).split()}
    return sides.pop() if len(sides) == 1 else "Unknown"


def normalise(names):
    return names.str.replace('"', "").str.replace(r"\s+", " ", regex=True).str.strip().str.lower()


def kaggle_regression(source):
    """The 891 labelled Kaggle passengers must appear in titanic3 with identical labels."""
    kaggle = pd.read_csv(BytesIO(fetch(KAGGLE_SOURCE, KAGGLE_SHA256)))
    key = lambda names, tickets: normalise(names) + "|" + tickets.astype(str).str.strip()
    source_keys = key(source.name, source.ticket)
    if source_keys.duplicated().any():
        raise ValueError("titanic3 name/ticket keys are not unique")
    lookup = pd.Series(range(len(source)), index=source_keys)
    rows = lookup.reindex(key(kaggle.Name, kaggle.Ticket))
    if rows.isna().any():
        raise ValueError("Some Kaggle passengers are missing from titanic3")
    matched = source.iloc[rows.astype(int).to_numpy()]
    for theirs, ours in [("Survived", "survived"), ("Sex", "sex"), ("Pclass", "pclass")]:
        if not np.array_equal(kaggle[theirs].to_numpy(), matched[ours].to_numpy()):
            raise ValueError(f"Kaggle and titanic3 disagree on {theirs}")
    flags = np.zeros(len(source), dtype=int)
    flags[rows.astype(int).to_numpy()] = 1
    return flags, {
        "kaggle_source_url": KAGGLE_SOURCE, "kaggle_source_sha256": KAGGLE_SHA256,
        "kaggle_rows_matched": int(len(kaggle)),
        "kaggle_fields_identical": ["Survived", "Sex", "Pclass"],
        "kaggle_survived": int(kaggle.Survived.sum()),
    }


def grouped_splits(frame, groups):
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train, test = next(outer.split(frame, frame.Survived, groups))
    split = pd.DataFrame({"PassengerId": frame.PassengerId, "subset": "train",
                          "cv_fold": -1, "ticket_group": groups})
    split.loc[test, "subset"] = "test"
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
    for fold, (_, validation) in enumerate(inner.split(
            frame.iloc[train], frame.Survived.iloc[train], groups.iloc[train])):
        split.loc[train[validation], "cv_fold"] = fold
    if set(split.ticket_group[split.subset.eq("train")]) & set(split.ticket_group[split.subset.eq("test")]):
        raise AssertionError("Ticket groups overlap the holdout boundary")
    repeated = []
    for repeat in range(REPEATS):
        splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True,
                                        random_state=SEED + 100 + repeat)
        fold_of = np.full(len(frame), -1)
        for fold, (_, held) in enumerate(splitter.split(frame, frame.Survived, groups)):
            fold_of[held] = fold
        repeated.append(pd.DataFrame({"PassengerId": frame.PassengerId,
                                      "repeat": repeat, "fold": fold_of}))
    repeated = pd.concat(repeated, ignore_index=True)
    crossing = repeated.assign(group=np.tile(groups.to_numpy(), REPEATS)).groupby(
        ["repeat", "group"]).fold.nunique()
    if crossing.max() != 1 or (repeated.fold < 0).any():
        raise AssertionError("Repeated folds split a ticket group or left rows unassigned")
    return split, repeated, train, test


def simulate_promotions(n=5000, seed=SEED):
    """Synthetic promotion history with known, injected bias (no real employees).

    True process: promotion depends on latent merit, plus a -0.70 log-odds penalty for
    women (written-policy analogue) and -0.50 for the regional site (structural access).
    Part-time status has NO effect on promotion; it is a proxy correlated with gender.
    The merit-only counterfactual uses the same random draws with both penalties removed.
    """
    rng = np.random.default_rng(seed)
    female = rng.random(n) < .48
    regional = rng.random(n) < .35
    part_time = rng.random(n) < np.where(female, .30, .04)
    tenure = np.round(rng.gamma(2.2, 2.4, n), 1)
    merit = rng.normal(size=n)
    rating = np.round(np.clip(3 + .8 * (merit + rng.normal(scale=.6, size=n)), 1, 5), 2)
    tenure_z = (tenure - tenure.mean()) / tenure.std()
    base = -1.6 + 1.3 * merit + .15 * tenure_z
    u = rng.random(n)
    historical = u < 1 / (1 + np.exp(-(base - .7 * female - .5 * regional)))
    merit_only = u < 1 / (1 + np.exp(-base))
    frame = pd.DataFrame({
        "EmployeeId": np.arange(1, n + 1), "gender": np.where(female, "Woman", "Man"),
        "site": np.where(regional, "Regional", "Headquarters"),
        "part_time": part_time.astype(int), "tenure_years": tenure, "rating": rating,
        "promoted": historical.astype(int), "promoted_merit_only": merit_only.astype(int),
        "subset": np.where(rng.random(n) < .7, "train", "test"),
    })
    truth = {
        "n": n, "seed": seed, "synthetic": True,
        "process": "logit P(promoted) = -1.6 + 1.3*merit + 0.15*tenure_z - 0.70*woman - 0.50*regional",
        "injected_log_odds": {"woman": -0.70, "regional": -0.50, "part_time": 0.0},
        "observed_rating": "clip(3 + 0.8*(merit + N(0, 0.6^2)), 1, 5), rounded to 0.01",
        "part_time_probability": {"Woman": .30, "Man": .04},
        "counterfactual": "promoted_merit_only reuses the same uniform draws with both penalties set to 0",
        "warning": "Synthetic, illustrative data. No real employees or organisations.",
    }
    return frame, truth


def main():
    raw = fetch(SOURCE, EXPECTED_SHA256)
    source = pd.read_csv(BytesIO(raw))
    if len(source) != 1309:
        raise ValueError("The pinned titanic3 source no longer has 1,309 rows")
    in_kaggle, regression = kaggle_regression(source)
    tickets = source.ticket.astype(str).str.strip()
    ticket_size = tickets.map(tickets.value_counts())
    frame = pd.DataFrame({
        "PassengerId": np.arange(1, len(source) + 1),
        "Survived": source.survived.astype(int), "Pclass": source.pclass.astype(int),
        "Sex": source.sex, "Age": source.age, "SibSp": source.sibsp, "Parch": source.parch,
        "Fare": source.fare, "TicketSize": ticket_size.astype(int),
        "FarePerPerson": source.fare / ticket_size, "Embarked": source.embarked,
        "Title": source.name.map(title_group), "BoatSide": source.boat.map(boat_side),
        "InKaggleTrain": in_kaggle,
    })
    group_numbers = {t: f"G{i:04d}" for i, t in enumerate(sorted(tickets.unique()))}
    groups = tickets.map(group_numbers)
    split, repeated, train, test = grouped_splits(frame, groups)
    promotions, truth = simulate_promotions()
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    frame.to_csv(data / "passengers.csv", index=False, float_format="%.10g")
    split.to_csv(data / "split.csv", index=False)
    repeated.to_csv(data / "repeated_cv.csv", index=False)
    promotions.to_csv(data / "synthetic_promotions.csv", index=False, float_format="%.10g")
    (data / "synthetic_truth.json").write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "source_url": SOURCE, "source_sha256": sha256(raw).hexdigest(),
        "source_name": "titanic3 (Thomas Cason, from Encyclopedia Titanica, state of 2 August 1999)",
        "source_page": "https://hbiostat.org/data/repo/titanic.html",
        "terms": "https://hbiostat.org/data/ - 'Permission is granted to anyone wishing to use the data sets provided here.'",
        "required_credit": CREDIT,
        "upstream_note": ("Built from Encyclopedia Titanica, whose own content is not necessarily "
                          "public domain; this repository therefore commits no names or biographical text."),
        "rows": len(frame), "survived": int(frame.Survived.sum()),
        "coverage": "Passengers only (crew excluded); not the complete ship's company",
        **regression,
        "split_seed": SEED, "split_method": "StratifiedGroupKFold; first outer fold held out",
        "grouping": "Shared ticket strings, replaced by invented group identifiers",
        "train_rows": len(train), "test_rows": len(test),
        "train_survived": int(frame.Survived.iloc[train].sum()),
        "test_survived": int(frame.Survived.iloc[test].sum()),
        "repeated_cv": f"{REPEATS} repeats x {FOLDS} StratifiedGroupKFold folds over all rows; seeds {SEED + 100}..{SEED + 100 + REPEATS - 1}",
        "derived_columns": {
            "TicketSize": "Rows in titanic3 sharing the ticket string",
            "FarePerPerson": "Fare / TicketSize (Fare is recorded per ticket, not per person)",
            "Title": "Mr / Mrs (incl. Mme) / Miss (incl. Mlle, Ms) / Master / Rare; name discarded",
            "BoatSide": "Recorded boat mapped to Starboard (odd, A, C) or Port (even, B, D); descriptive only, never a predictor",
            "InKaggleTrain": "1 if the passenger is in the 891-row Kaggle training sample",
        },
        "excluded_at_ingest": ["name", "ticket", "cabin", "boat", "body", "home.dest"],
    }
    (data / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
