"""Accessible static exhibits drawn from actual committed records and results."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from .analysis import MODELS, reliability

COLORS = ["#60788a", "#007d80", "#c46632"]
INK = "#183144"


def style():
    plt.rcParams.update({
        "figure.facecolor": "#f6f3ec", "axes.facecolor": "#fffdf8",
        "axes.edgecolor": "#c5cece", "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titleweight": "bold", "axes.spines.top": False,
        "axes.spines.right": False, "grid.alpha": .18,
    })


def eda_figure(data):
    style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, variable, title in zip(
        axes.flat[:2], ["Sex", "Pclass"], ["Survival by recorded sex", "Survival by passenger class"]
    ):
        grouped = data.groupby(variable).Survived.agg(["mean", "size"])
        ax.bar(grouped.index.astype(str), grouped["mean"], color=COLORS[1])
        for i, (_, row) in enumerate(grouped.iterrows()):
            ax.text(i, row["mean"] + .025, f'n={int(row["size"])}', ha="center", fontsize=9)
        ax.set(title=title, xlabel=variable, ylabel="Observed survival proportion", ylim=(0, 1))
    for ax, variable, bins in [(axes[1, 0], "Age", 20), (axes[1, 1], "FarePerPerson", 25)]:
        bin_edges = np.histogram_bin_edges(data[variable].dropna(), bins=bins)
        for outcome, color, label in [(0, COLORS[0], "Did not survive"), (1, COLORS[2], "Survived")]:
            ax.hist(data.loc[data.Survived.eq(outcome), variable].dropna(), bins=bin_edges,
                    alpha=.55, color=color, label=label)
        ax.set(title=f"{variable}: observed values only", xlabel=variable, ylabel="Passenger records")
        ax.legend(frameon=False)
    fig.suptitle("The labeled sample: visible patterns, incomplete history", fontsize=18, fontweight="bold")
    return fig


def diagnostics(tables):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for model, color in zip(MODELS, COLORS):
        group = tables["predictions"].loc[lambda d: d.model.eq(model)]
        fpr, tpr, _ = roc_curve(group.Survived, group.probability, pos_label=1)
        axes[0].plot(fpr, tpr, color=color, label=model)
        cal = reliability(group.Survived, group.probability)
        visible = cal.loc[cal["n"].gt(0)]
        axes[1].plot(visible.predicted, visible.observed, "o-", color=color, label=model)
        for row in visible.itertuples():
            axes[1].annotate(f"n={row.n}", (row.predicted, row.observed),
                             xytext=(4, 5), textcoords="offset points", fontsize=8, color=color)
    for ax in axes:
        ax.plot([0, 1], [0, 1], "--", color="#9aa5ad", linewidth=1)
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.grid()
    axes[0].set(title="Held-out ROC | event = survived", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="Reliability | five equal-width bins", xlabel="Mean predicted survival probability",
                ylabel="Observed survival proportion")
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    n_test = len(tables["predictions"]) // len(MODELS)
    fig.suptitle(f"{n_test} untouched records, not a final verdict", fontsize=17, fontweight="bold")
    return fig


def effects_figure(tables):
    style()
    coefficients = tables["coefficients"].iloc[1:].iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    y = np.arange(len(coefficients))
    ax.errorbar(coefficients.estimate, y,
                xerr=[coefficients.estimate - coefficients.lower, coefficients.upper - coefficients.estimate],
                fmt="o", color=COLORS[1], capsize=3)
    ax.axvline(0, linestyle="--", color=COLORS[0])
    ax.set(yticks=y, yticklabels=coefficients.term, xlabel="Log-odds coefficient",
           title="Bayesian logistic | approximate marginal 95% credible intervals")
    fig.suptitle("Associations after adjustment are still not causes", fontsize=16, fontweight="bold")
    return fig


def overview(data, tables, path):
    style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    rates = tables["bayesian_rates"]
    for sex, color, offset in [("female", COLORS[1], -.12), ("male", COLORS[2], .12)]:
        group = rates.loc[rates.sex.eq(sex)]
        axes[0, 0].errorbar(group.pclass + offset, group.posterior_mean,
                            yerr=[group.posterior_mean - group.lower, group.upper - group.posterior_mean],
                            fmt="o", capsize=4, color=color, label=sex)
        for row in group.itertuples():
            axes[0, 0].annotate(f"n={row.n}", (row.pclass + offset, row.posterior_mean),
                                xytext=(3, 7), textcoords="offset points", fontsize=8)
    axes[0, 0].set(title="01 / Describe the observed sample", xlabel="Passenger class",
                   ylabel="Survival probability | Beta(1,1) posterior", xticks=[1, 2, 3], ylim=(0, 1.08))
    axes[0, 0].legend(frameon=False)
    rules = tables["rules"].head(5).iloc[::-1]
    labels = [r.antecedent.replace(" & ", "\n") + "\n→ " + r.consequent.removeprefix("Outcome=")
              for r in rules.itertuples()]
    axes[0, 1].barh(np.arange(len(rules)), rules.lift, color=COLORS[1])
    axes[0, 1].set(yticks=np.arange(len(rules)), yticklabels=labels,
                   xlabel="Lift over training outcome baseline", title="02 / Discover training-only rules")
    axes[0, 1].tick_params(axis="y", labelsize=8)
    axes[0, 1].axvline(1, linestyle="--", color=COLORS[0])
    test = tables["metrics"].query("phase == 'test'").set_index("model").loc[MODELS]
    cv = tables["metrics"].query("phase == 'cv'").groupby("model").roc_auc.agg(["mean", "std"]).loc[MODELS]
    x = np.arange(3)
    axes[1, 0].bar(x - .18, cv["mean"], width=.34, color=COLORS[0], label="Mean fold validation")
    axes[1, 0].bar(x + .18, test.roc_auc, width=.34, color=COLORS[1], label="Untouched test")
    axes[1, 0].set(xticks=x, xticklabels=["Prevalence", "Random forest", "Bayesian logistic"],
                   ylim=(0, 1), ylabel="ROC AUC | event = survived", title="03 / Evaluate without retuning")
    axes[1, 0].legend(frameon=False)
    coefficients = tables["coefficients"].iloc[1:].iloc[::-1]
    axes[1, 1].errorbar(coefficients.estimate, np.arange(len(coefficients)),
                       xerr=[coefficients.estimate - coefficients.lower, coefficients.upper - coefficients.estimate],
                       fmt="o", capsize=3, color=COLORS[2])
    axes[1, 1].axvline(0, linestyle="--", color=COLORS[0])
    axes[1, 1].set(yticks=np.arange(len(coefficients)), yticklabels=coefficients.term,
                   xlabel="Log odds | approximate marginal 95% intervals", title="04 / Quantify model uncertainty")
    fig.suptitle("TITANIC / Patterns, probabilities, and the limits of hindsight\n"
                 f"{len(data):,} labeled records · {data.subset.eq('train').sum()} training · "
                 f"{data.subset.eq('test').sum()} held out · public historical data",
                 fontsize=20, fontweight="bold", color=INK)
    fig.supxlabel("Selected historical records, not the full manifest. Associations are not causal effects; "
                  "this is not a rescue-prioritization tool.", fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    for name, figure in [("eda.png", eda_figure(data)), ("roc_calibration.png", diagnostics(tables)),
                         ("bayesian_effects.png", effects_figure(tables))]:
        figure.savefig(path.parent / name, dpi=150, bbox_inches="tight")
        plt.close(figure)
