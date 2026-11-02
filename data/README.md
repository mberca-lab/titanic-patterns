# Public historical data

## titanic3

- **Source:** `titanic3` from the [Vanderbilt Department of Biostatistics data
  repository](https://hbiostat.org/data/repo/titanic.html), compiled by Thomas
  Cason from Encyclopedia Titanica.
- **Records:** 1,309 passengers; crew are not included.
- **Survivors:** 500.

> Data obtained from http://hbiostat.org/data courtesy of the Vanderbilt
> University Department of Biostatistics.

The hbiostat terms state: "Permission is granted to anyone wishing to use the
data sets provided here." Upstream Encyclopedia Titanica content is not
necessarily public domain. For that reason, names, tickets, cabins, boat
numbers, body numbers and home/destination text are **dropped at ingest** and
never committed.

The 891-row Kaggle training sample (Paul Hendricks' CC0 `titanic` package) is
kept as a **regression test**. Every Kaggle row matches titanic3 exactly on
survival, sex and class, and `InKaggleTrain` flags those rows.

## Files

| File | Purpose |
|---|---|
| [`passengers.csv`](passengers.csv) | Minimal shared input for R and Python (1,309 rows) |
| [`split.csv`](split.csv) | Ticket-grouped holdout (1,048 train / 261 test) and inner folds 0–4 |
| [`repeated_cv.csv`](repeated_cv.csv) | 20 repeats × 5 ticket-grouped folds over all rows |
| [`synthetic_promotions.csv`](synthetic_promotions.csv) | **Synthetic** 5,000-employee promotion data with a known, injected bias |
| [`synthetic_truth.json`](synthetic_truth.json) | The injected log-odds and the generating process |
| [`provenance.json`](provenance.json) | Source URLs, SHA-256 checksums, terms, credit, excluded fields and derived columns |

### Columns in `passengers.csv`

| Column | Meaning |
|---|---|
| `PassengerId` | Join key only |
| `Survived` | 1 = survived |
| `Pclass` | Passenger class: 1, 2 or 3 |
| `Sex` | Source labels, female or male |
| `Age` | Missing values retained |
| `SibSp`, `Parch` | Siblings/spouses and parents/children aboard |
| `Fare` | Recorded per ticket |
| `TicketSize` | Rows sharing the ticket |
| `FarePerPerson` | `Fare / TicketSize` |
| `Embarked` | C, Q or S; missing values retained |
| `Title` | Mr, Mrs, Miss, Master or Rare. The name itself is discarded |
| `BoatSide` | Starboard, Port, Unknown or None recorded; descriptive only |
| `InKaggleTrain` | 1 if the row is in the Kaggle sample |

Ticket strings are used once, to build invented group identifiers, and are then
discarded. Groups do not perfectly identify families. Neither group IDs nor
folds are predictors.

The synthetic promotion data are **not real employees**. They exist so that the
audit can be checked against a known truth.

## Recreate the data

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py
```

This is the only step that downloads anything. Both sources are pinned by
checksum, and the script fails if either file changes or the Kaggle regression
breaks.

Read the [analysis contract](../METHODOLOGY.md) before interpreting the results.
