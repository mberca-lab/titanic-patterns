"""Build the narrative notebook from reusable analysis functions, without fitting."""

import nbformat as nbf
from .analysis import ROOT


def make_notebook():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    notebook = nbf.v4.new_notebook()
    notebook.metadata.kernelspec = {
        "display_name": "Python 3 (Titanic)", "language": "python", "name": "python3",
    }
    notebook.cells = [
        md("# TITANIC\n## Patterns in the record: prediction is not explanation\n\n"
           "**Question:** What do descriptive patterns, predictive comparisons and uncertainty tell us "
           "about historical passenger records? The models do not identify a decision policy or its "
           "causal effect. A separate synthetic promotion example tests selected audit methods under "
           "specified generating assumptions, not in real workplaces.\n\n"
           "This is a historical observational case study, not a rescue-prioritization system. "
           "No individual simulation or causal claim is offered."),
        md("### Decision log / public data, bounded scope\n\n"
           "- Source: committed `data/passengers.csv`, a minimal extract of titanic3. Data obtained from "
           "http://hbiostat.org/data courtesy of the Vanderbilt University Department of Biostatistics.\n"
           "- Coverage: **1,309 passengers**; crew are not included. The 891-row Kaggle sample is a regression test.\n"
           "- Level: standard EDA plus association discovery and prespecified predictive analysis.\n"
           "- Protocol: seed 20261014, committed ticket-grouped split and five training folds. "
           "1,048 training records, 261 held out, plus 20 x 5 repeated grouped folds for the audit. "
           "No tuning on the test set.\n"
           "- Fare is divided by ticket size (FarePerPerson). "
           "Original names, tickets and cabin strings are not loaded; group labels and IDs are never predictors.\n"
           "- Every analytical computation below calls reusable code in `Python/`."),
        code("from pathlib import Path\nimport sys\n"
             "root = Path.cwd()\nif not (root / 'data' / 'passengers.csv').exists():\n    root = root.parent\n"
             "sys.path.insert(0, str(root))\n"
             "import pandas as pd\nimport matplotlib.pyplot as plt\n"
             "from IPython.display import display, Image\n"
             "from Python.analysis import load_data, missingness, run_analysis, MODELS\n"
             "from Python.figures import eda_figure\n"
             "pd.set_option('display.max_columns', 20)\n"
             "data = load_data(root)\n"
             "print(f'{len(data):,} records | {data.PassengerId.nunique():,} unique passenger IDs')\n"
             "display(data.drop(columns=['ticket_group']).head())\n"
             "display(data.groupby('subset').Survived.agg(['size', 'sum', 'mean']))"),
        md("## 01 / First understand what was recorded\n\n"
           "Missing ages are retained as unknown in association baskets and imputed only within model-training "
           "partitions. Fare and age distributions are not symmetric. Missing records and the labeled-sample "
           "selection constrain interpretation; outliers are not automatically errors."),
        code("display(missingness(data))\n"
             "print('Exact duplicate records:', data.duplicated().sum())\n"
             "display(data[['Age', 'FarePerPerson', 'SibSp', 'Parch']].describe())\n"
             "display(data[['Pclass', 'Age', 'FarePerPerson', 'SibSp', 'Parch', 'Survived']].corr(method='spearman'))\n"
             "figure = eda_figure(data)\ndisplay(figure)\nplt.close(figure)"),
        md("## 02 / Describe uncertainty before predicting\n\n"
           "Within each sex-by-class group, start with Beta(1,1), add observed survived and non-survived counts, "
           "and calculate the exact central 95% beta credible interval.            These summaries use **all 1,309 "
           "records** and are separate from predictive evaluation. They describe this selected sample, not "
           "causal sex or class effects in a population."),
        code("data, tables, summary = run_analysis(root)\n"
             "display(tables['bayesian_rates'])\n"
             "print('Survived:', summary['survived'], 'of', summary['n'])"),
        md("## 03 / Discover transparent training-only associations\n\n"
           "Apriori considers itemsets of at most three items, with survival only as a singleton consequent. "
           "Support ≥ .04, confidence ≥ .60, joint count ≥ 20 and lift ≥ 1 are fixed exploratory filters. "
           "For each rule we show counts and its training-population outcome baseline. A high lift is an "
           "association, not a significance test, causal explanation or an individual prediction."),
        code("print('Retained rules:', len(tables['rules']))\n"
             "display(tables['rules'].head(12))"),
        md("## 04 / An honest comparison: grouped validation, then an untouched holdout\n\n"
           "All three models use the same committed five folds. Each fold relearns age/fare medians, "
           "alphabetically tie-broken embarkation mode and continuous-feature sample-SD scaling. "
           "The prevalence baseline also refits inside each fold.\n\n"
           "The random forest uses 500 trees, max_features=3, min_samples_leaf=5, seed 20261014, "
           "two workers and no tuning. Bayesian logistic uses independent Normal(0,5²) intercept "
           "and Normal(0,2.5²) coefficient priors. Test predictions are posterior-mode plug-ins, "
           "not integrated posterior predictions.\n\n"
           "ROC event = survived; no ROC reversal. Balanced accuracy uses probability ≥ .5. "
           "Lower Brier and log loss are better. Fold SD is descriptive variability, not a confidence interval."),
        code("display(tables['metrics'].query(\"phase == 'cv'\").groupby('model')["
             "['roc_auc', 'brier', 'log_loss', 'balanced_accuracy']].agg(['mean', 'std']))\n"
             "display(tables['metrics'].query(\"phase == 'test'\"))\n"
             "display(Image(filename=str(root / 'outputs' / 'python' / 'roc_calibration.png')))\n"
             "display(tables['calibration'])"),
        md("## 05 / Bayesian effects: uncertainty is part of the result\n\n"
           "The inverse analytic negative-log-posterior Hessian provides the local Laplace covariance. "
           "Displayed intervals are **approximate marginal 95% credible intervals**, not exact intervals "
           "or MCMC output. Continuous effects are per one training sample SD (fare is log1p transformed). "
           "Reference levels are female, first class and embarkation at C. Odds ratios are not risk ratios. "
           "The coefficient intervals do not include model-selection or dataset-selection uncertainty."),
        code("display(tables['coefficients'])\n"
             "print('Posterior max absolute gradient:', summary['bayesian_max_abs_gradient'])\n"
             "print('Optimizer:', summary['bayesian_optimizer_message'])\n"
             "display(Image(filename=str(root / 'outputs' / 'python' / 'bayesian_effects.png')))"),
        md("## 07 / Recorded groups and historical context\n\n"
           "Women and children vs men by class, sensitivity to the child cutoff, the side of the ship each "
           "lifeboat was launched from, and where ages are missing. All proportions carry Wilson 95% intervals. "
           "Boat side is descriptive only and never a predictor."),
        code("for name in ['policy_groups', 'child_cutoffs', 'boat_side', 'missing_age']:\n"
             "    print(name)\n    display(tables[name])"),
        md("## 08 / What does a model learn? Repeated grouped CV with paired intervals\n\n"
           "Policy-only, structure-only and combined Laplace logistic models, plus sex-blind and sex-blind + "
           "title variants and a random-forest benchmark, all scored on the same 20 x 5 ticket-grouped folds. "
           "Differences vs the reference (Policy + structure) use Nadeau-Bengio corrected t intervals."),
        code("display(tables['policy_fit'])\n"
             "display(tables['model_comparison'].query(\"metric == 'roc_auc'\")"
             "[['model', 'mean_model', 'mean_difference', 'lower', 'upper']])"),
        md("## 09 / Attribute removal, proxies and group diagnostics\n\n"
           "Dropping sex terms removes most of the predicted sex gap, but adding title (Mr/Mrs/Miss/Master) "
           "brings it back: title is a near-perfect proxy for recorded sex. This does not identify a "
           "historical mechanism or establish discrimination. Group diagnostics use out-of-fold "
           "probabilities: rate ratios, agreement in group averages and error rates, with cells below 20 "
           "suppressed. Average agreement does not establish calibration across the probability range."),
        code("display(tables['proxy_gap'])\ndisplay(tables['sex_recovery'])\n"
             "display(tables['fairness_audit'][['grouping', 'group', 'n', 'base_rate', 'mean_predicted', "
             "'selection_rate', 'impact_ratio', 'ir_lower', 'ir_upper', 'tpr', 'fpr', 'ppv']])"),
        md("## 10 / A separate synthetic promotion example\n\n"
           "5,000 **synthetic** employees with an injected -0.70 log-odds penalty for women, -0.50 for "
           "regional staff and no part-time effect. A gender-visible model recovers the penalty; a "
           "gender-blind model moves it onto part-time work, a proxy with no true effect."),
        code("display(tables['sim_coefficients'])\ndisplay(tables['sim_gap'])\n"
             "display(tables['sim_recovery'])\ndisplay(tables['sim_impact'])"),
        md("## 11 / A story with limits, not a leaderboard\n\n"
           "The descriptive sex/class differences, rule associations and adjusted model coefficients answer "
           "different questions. They need not agree in magnitude. The small held-out sample and dependent "
           "historical records make score differences uncertain: no universal winner or statistically "
           "significant gain is claimed. Random-forest predictions may differ across R and Python, whereas "
           "the committed design matrices and Bayesian objective should match numerically.\n\n"
           "**Fairness and selection:** historically recorded sex and class are not grounds for discriminatory "
           "decisions. Curated labeled records are not a random sample of a modern deployment population. "
           "Ticket groups imperfectly approximate family/social dependence. Unknown ages, incomplete records "
           "and correlated predictors limit generalization. These results cannot determine why an individual "
           "survived and must not support rescue recommendations."),
        code("display(Image(filename=str(root / 'outputs' / 'python' / 'overview.png')))\n"
             "display(pd.Series(summary, name='Reproducibility summary'))"),
    ]
    destination = ROOT / "notebooks" / "01_titanic_python.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, destination)
    print(destination)


if __name__ == "__main__":
    make_notebook()
