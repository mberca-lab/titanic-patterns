# Titanic / The Rule in the Data

### What a model learns when it is trained on a historical decision

On 15 April 1912 about 1,500 people died when the Titanic sank. Who reached a
lifeboat was shaped by a human rule, *women and children first*, which was
applied unevenly by class and by side of the ship.

This project treats that record the way a **people-analytics team should treat
past promotion, hiring or performance decisions**. Train a model on historical
outcomes and it learns the decision rule that produced them, not merit or worth.
Remove the protected attribute and a proxy can carry the rule straight back in.

**A real, public-data case study in R and Python.** It covers 1,309 titanic3
passengers, uses repeated grouped cross-validation, and runs an HR-style
fairness audit. A synthetic promotion simulation with a known, injected bias
checks whether the audit catches what we planted.

![The Rule in the Data: memorial hero and the first chart, women and children vs men by class](outputs/story-preview.png)

**[Explore the interactive story](https://mberca.github.io/titanic-patterns/)** ·
**[Start in R](R/README.md)** · **[Python edition](Python/README.md)** ·
**[Methodology](METHODOLOGY.md)** · **[Data guide](data/README.md)**

> Not a survival game. No individual names are published, and nothing here
> scores or ranks people. Associations are not causes.

## Five findings

| # | Question | What the evidence shows |
|---|---|---|
| 1 | **Was there a rule?** | Women and children: 70% survived. Men: 17%. The rule weakened with class: 97% in first class, 46% in third. |
| 2 | **Was it applied evenly?** | Among recorded lifeboat occupants, adult men made up **41%** of starboard boats but **12%** of port boats. The officers on each side read the same order differently. |
| 3 | **What does a model learn?** | The policy terms alone reach a cross-validated AUC of 0.79. Ship structure alone (class, fare, port) reaches 0.69. Together they reach **0.85**. A random forest adds nothing reliable (difference 0.00, 95% CI −0.01 to 0.01). |
| 4 | **Does removing sex remove the bias?** | Without sex terms the model reproduces 14% of the observed sex gap. Add the passenger's **title** and it reproduces **98%**. Title recovers recorded sex with AUC 0.9999. |
| 5 | **Would an HR audit catch it?** | The impact ratio for men is 0.035, far below the four-fifths heuristic. The model is calibrated within each group, yet false-positive rates differ: 0.57 for women vs 0.006 for men. Both cannot be equalised at once. |

### If this were promotion data

The synthetic simulation contains 5,000 employees and is clearly labelled as
synthetic. It injects a −0.70 log-odds penalty for women and a −0.50 penalty for
regional staff. Part-time status has **no** effect.

- **Historical decisions:** promotion rates are 12.4% for women and 21.4% for
  men, an impact ratio of 0.58.
- **Merit-only counterfactual:** the rates become 23.2% vs 23.6%, an impact
  ratio of 0.98.
- **Gender-visible model:** it recovers the planted penalty, −0.68 with a 95% CI
  of −0.89 to −0.46.
- **Gender-blind model:** the predicted gap shrinks, but the model now
  penalises **part-time work** (−0.29, CI excludes 0), which had no true effect.
  The reason is that 29% of women and 4% of men in the simulation are part-time.

The bias did not disappear. It moved to a proxy.

## Find your way

| Your goal | Start here |
|---|---|
| See the visual story without installing anything | [Interactive story](https://mberca.github.io/titanic-patterns/) |
| Run or read the R edition | [`R/`](R/) · [`titanic-patterns.Rproj`](titanic-patterns.Rproj) |
| Run or read the Python edition | [`Python/`](Python/) · [Executed notebook](notebooks/01_titanic_python.ipynb) |
| Understand where the data came from | [Data guide](data/README.md) · [Pinned provenance](data/provenance.json) |
| Inspect splits and repeated folds | [`data/split.csv`](data/split.csv) · [`data/repeated_cv.csv`](data/repeated_cv.csv) |
| Inspect the audit tables | [Python results](outputs/python/) · [R results](outputs/r/) |
| Check what was prespecified | [Analysis contract](METHODOLOGY.md) |

## What makes this more than a Titanic accuracy exercise

- **A decision lens.** Policy-only, structure-only and combined models separate
  the rule from the ship. Sex-blind and proxy tests show how the rule survives
  removal.
- **Honest uncertainty.** The models run 20 × 5 repeated cross-validation,
  grouped by shared ticket. Model differences use paired **Nadeau–Bengio**
  corrected intervals rather than one lucky split.
- **An audit an HR team would recognise:**
  - impact ratios with Katz intervals;
  - per-group calibration;
  - TPR, FPR and PPV with Wilson intervals;
  - suppression of any cell with a denominator below 20.
- **A known-truth check.** The synthetic simulation tests whether the methods
  recover a bias we planted ourselves.
- **Two languages, one answer.** R and Python share every split and fold, and a
  parity script checks that the Bayesian fits and audit tables agree. The random
  forests are deliberately excluded, because their implementations differ.
- **Careful data work:**
  - fare is divided by ticket size, because it was recorded per ticket;
  - the 891-row Kaggle sample is kept as a regression test;
  - no names, tickets, cabins or biographical text are committed.

The original analytical lenses remain in the interactive appendix. Select an
association rule to compare its confidence with the same outcome's training
baseline, with both denominators visible. Highlight a model across the held-out
scorecard, ROC and calibration views without hiding the alternatives or changing
the evaluation.

## Run it

Use the [R guide](R/README.md) or [Python guide](Python/README.md) to install
dependencies and run the analysis. Normal runs read the committed extract and
need no internet access or account.

```powershell
.\.venv\Scripts\python.exe -m Python.run_analysis      # about 90 s, includes the audit
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts\check_parity.py     # after running the R edition
.\.venv\Scripts\python.exe scripts\build_story.py      # rebuild docs/index.html
```

R and Python write their results to `outputs/r/` and `outputs/python/` respectively.
The browser story is a single self-contained HTML file with no backend,
analytics, external fonts or CDN.

## Interpretation and responsible use

- This is an **observational** study of curated historical records. It covers
  passengers only, not crew. Ages are missing for 29% of third-class passengers
  and 6–12% of other classes, which is a data-quality gap that falls unevenly.
- The source records sex as a binary.
- Coefficients and rules describe associations within this record. They do not
  explain anyone's fate.
- The four-fifths rule and the calibration/error-rate impossibility are cited as
  **context for analysts**, not legal advice.
- The simulation is synthetic by design. Its value is that the truth is known.
- Read the [full methodology](METHODOLOGY.md) before reusing any part of this.

## Repository map

```text
R/                       R edition: analysis, audit, tests, narrative
Python/                  Python reference: analysis.py, audit.py, figures, notebook builder
notebooks/               Executed Python notebook
data/                    titanic3 extract, splits, repeated folds, synthetic promotions, provenance
outputs/python, outputs/r Model scores, audit tables, intervals, figures
scripts/prepare_data.py  Recreate and checksum-check every data file
scripts/check_parity.py  Compare shared evidence across R and Python
scripts/build_story.py   Bundle result tables into the offline story
web/story.html           Authored story layout and interactions
docs/index.html          Generated standalone story (GitHub Pages)
METHODOLOGY.md           Prespecified features, models, audit rules and limits
```

## Data credit and licence

Data obtained from http://hbiostat.org/data courtesy of the Vanderbilt University
Department of Biostatistics. titanic3 was compiled by Thomas Cason from
Encyclopedia Titanica. The hbiostat terms state: "Permission is granted to anyone
wishing to use the data sets provided here." Upstream content is not necessarily
public domain, so this repository commits only the minimal fields listed in
[provenance](data/provenance.json). It contains no names or biographical text.

Historical context comes from the British Wreck Commissioner's Inquiry (1912):
1,178 lifeboat seats and about 711 people saved.

Application code and original documentation are released under the
[MIT licence](LICENSE). The generated story embeds the Instrument Serif font under
the SIL Open Font License (see `web/assets/`). All other dependencies keep their own licences.
