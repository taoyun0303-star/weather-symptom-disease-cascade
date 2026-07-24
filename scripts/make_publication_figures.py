from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE = PROJECT_ROOT / "tmp" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from weather_health.constants import MODELED_SYMPTOMS


RESULT_DIR = PROJECT_ROOT / "results" / "publication"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "publication"
FIGURE_DIR = PROJECT_ROOT / "figures" / "publication"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#2F5D8C"
BLUE_LIGHT = "#9DB7D1"
GOLD = "#C5962D"
ORANGE = "#D47B35"
PINK = "#B7657B"
CHARCOAL = "#263238"
GREY = "#89939B"
LIGHT_GREY = "#E8ECEF"
WHITE = "#FFFFFF"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "axes.edgecolor": CHARCOAL,
        "axes.linewidth": 0.8,
        "axes.facecolor": WHITE,
        "figure.facecolor": WHITE,
        "xtick.color": CHARCOAL,
        "ytick.color": CHARCOAL,
        "text.color": CHARCOAL,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(
        FIGURE_DIR / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor=WHITE,
    )
    fig.savefig(
        FIGURE_DIR / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=WHITE,
    )
    plt.close(fig)


def endpoint_accuracy_figure(endpoint: pd.DataFrame) -> None:
    labels = {
        "oracle_true_symptom_tabnet": "Oracle: true symptoms -> TabNet",
        "direct_xgb_history": "Context -> XGBoost",
        "skip_context_symptom_xgb": "Context + predicted symptoms -> XGBoost",
        "cascade_context_xgb_prob_xgb": "Context -> symptoms -> XGBoost",
        "cascade_context_xgb_prob_tabnet": "Context -> symptoms -> TabNet",
        "direct_mlp": "Weather -> MLP",
        "direct_dummy": "Class-prior baseline",
        "cascade_xgb_prob_tabnet": "Weather -> symptoms -> TabNet",
    }
    colors = {
        "oracle_true_symptom_tabnet": GOLD,
        "direct_xgb_history": BLUE,
        "skip_context_symptom_xgb": BLUE_LIGHT,
        "cascade_context_xgb_prob_xgb": BLUE_LIGHT,
        "cascade_context_xgb_prob_tabnet": ORANGE,
        "direct_mlp": GREY,
        "direct_dummy": LIGHT_GREY,
        "cascade_xgb_prob_tabnet": PINK,
    }
    selected = endpoint[endpoint["pipeline"].isin(labels)].copy()
    selected["label"] = selected["pipeline"].map(labels)
    selected = selected.sort_values("accuracy")
    xerr = np.vstack(
        [
            selected["accuracy"] - selected["accuracy_ci_low"],
            selected["accuracy_ci_high"] - selected["accuracy"],
        ]
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    bars = ax.barh(
        selected["label"],
        selected["accuracy"],
        xerr=xerr,
        color=[colors[p] for p in selected["pipeline"]],
        edgecolor=CHARCOAL,
        linewidth=0.7,
        capsize=3,
    )
    for bar, value in zip(bars, selected["accuracy"]):
        ax.text(
            value + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value * 100:.1f}%",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Accuracy")
    ax.set_title(
        "Locked-test disease classification accuracy", loc="left", pad=28
    )
    ax.text(
        0,
        1.01,
        "Group-disjoint test set, n=997; error bars show 95% bootstrap confidence intervals.",
        transform=ax.transAxes,
        fontsize=9,
        color=GREY,
    )
    ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.8)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    save_figure(fig, "Fig_Publication_Endpoint_Accuracy")


def stage1_figure(stage1: pd.DataFrame) -> None:
    labels = {
        "lr": "Weather LR",
        "mlp": "Weather MLP",
        "xgb": "Weather + indices XGBoost",
        "xgb_raw_weather": "Raw weather XGBoost",
        "xgb_pre_event_context": "Context XGBoost",
    }
    table = stage1[
        (stage1["scope"] == "locked_test") & stage1["model"].isin(labels)
    ].copy()
    table["label"] = table["model"].map(labels)
    table = table.sort_values("f1_micro")
    y = np.arange(len(table))

    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    height = 0.34
    ax.barh(
        y - height / 2,
        table["f1_micro"],
        height,
        label="Micro F1",
        color=BLUE,
        edgecolor=CHARCOAL,
        linewidth=0.6,
    )
    ax.barh(
        y + height / 2,
        table["f1_macro"],
        height,
        label="Macro F1",
        color=GOLD,
        edgecolor=CHARCOAL,
        linewidth=0.6,
    )
    ax.set_yticks(y, table["label"])
    ax.set_xlim(0, max(0.32, table[["f1_micro", "f1_macro"]].to_numpy().max() + 0.04))
    ax.set_xlabel("F1 score")
    ax.set_title("Stage-1 symptom prediction performance", loc="left", pad=28)
    ax.text(
        0,
        1.01,
        "Thirty-six estimable symptoms on the group-disjoint locked test set.",
        transform=ax.transAxes,
        fontsize=9,
        color=GREY,
    )
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.8)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    save_figure(fig, "Fig_Publication_Stage1_Performance")


def error_propagation_figure(endpoint: pd.DataFrame) -> None:
    labels = {
        "oracle_true_symptom_tabnet": "True\nsymptoms",
        "cascade_context_xgb_prob_tabnet": "Context-predicted\nsymptoms",
        "cascade_xgb_prob_tabnet": "Weather-predicted\nsymptoms",
    }
    table = endpoint.set_index("pipeline").loc[list(labels)].reset_index()
    table["label"] = table["pipeline"].map(labels)
    colors = [GOLD, ORANGE, PINK]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.bar(
        table["label"],
        table["accuracy"],
        color=colors,
        edgecolor=CHARCOAL,
        linewidth=0.7,
        width=0.64,
    )
    for bar, value in zip(bars, table["accuracy"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Disease accuracy")
    ax.set_title(
        "Disease performance by TabNet symptom representation",
        loc="left",
        pad=28,
    )
    ax.text(
        0,
        1.01,
        "Same disease-stage architecture; only the symptom representation changes.",
        transform=ax.transAxes,
        fontsize=9,
        color=GREY,
    )
    ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.8)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    save_figure(fig, "Fig_Publication_Error_Propagation")


def confusion_figure() -> None:
    matrix = pd.read_csv(RESULT_DIR / "confusion_matrices.csv")
    pipeline = "cascade_context_xgb_prob_xgb"
    table = matrix[matrix["pipeline"] == pipeline].pivot(
        index="true_class", columns="predicted_class", values="count"
    )
    row_order = sorted(table.index)
    table = table.loc[row_order, row_order]
    normalized = table.div(table.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    sns.heatmap(
        normalized,
        cmap=sns.light_palette(BLUE, as_cmap=True),
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.4,
        linecolor=WHITE,
        cbar_kws={"label": "Row-normalized proportion", "shrink": 0.75},
        ax=ax,
    )
    ax.set_title(
        "Confusion matrix for the context symptom-bottleneck model",
        loc="left",
        pad=28,
    )
    ax.text(
        0,
        1.005,
        "XGBoost symptom probabilities followed by XGBoost disease classification; n=997.",
        transform=ax.transAxes,
        fontsize=9,
        color=GREY,
    )
    ax.set_xlabel("Predicted disease")
    ax.set_ylabel("True disease")
    ax.tick_params(axis="x", rotation=55, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    save_figure(fig, "Fig_Publication_Context_Confusion")


def reliability_points(y_true: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == y_true
    bins = np.linspace(0, 1, 11)
    rows = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            rows.append(
                {
                    "confidence": float(confidence[mask].mean()),
                    "accuracy": float(correct[mask].mean()),
                    "count": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def calibration_figure() -> None:
    pipelines = {
        "direct_xgb_history": ("Context direct XGBoost", BLUE, "o"),
        "cascade_context_xgb_prob_xgb": (
            "Context symptom bottleneck",
            ORANGE,
            "s",
        ),
        "cascade_xgb_prob_tabnet": ("Weather symptom bottleneck", PINK, "^"),
        "oracle_true_symptom_tabnet": ("Oracle symptoms", GOLD, "D"),
    }
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot([0, 1], [0, 1], linestyle="--", color=CHARCOAL, label="Ideal")
    for pipeline, (label, color, marker) in pipelines.items():
        artifact = np.load(ARTIFACT_DIR / f"endpoint_{pipeline}.npz")
        points = reliability_points(
            artifact["test_true"], artifact["test_probabilities"]
        )
        ax.plot(
            points["confidence"],
            points["accuracy"],
            color=color,
            marker=marker,
            linewidth=1.7,
            markersize=5,
            label=label,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Locked-test reliability curves", loc="left", pad=28)
    ax.text(
        0,
        1.01,
        "Ten equal-width confidence bins; empty bins are omitted.",
        transform=ax.transAxes,
        fontsize=9,
        color=GREY,
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(color=LIGHT_GREY, linewidth=0.8)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    save_figure(fig, "Fig_Publication_Calibration")


def importance_figure() -> None:
    importance = pd.read_csv(RESULT_DIR / "feature_importance.csv")
    stage1 = (
        importance[importance["analysis"] == "stage1_context_xgb_micro_ap"]
        .nlargest(8, "importance_mean")
        .sort_values("importance_mean")
    )
    oracle = (
        importance[importance["analysis"] == "oracle_tabnet_mask"]
        .nlargest(8, "importance_mean")
        .sort_values("importance_mean")
    )
    stage1["feature"] = stage1["feature"].str.replace("_", " ", regex=False)
    oracle["feature"] = oracle["feature"].str.replace("_", " ", regex=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    axes[0].barh(
        stage1["feature"],
        stage1["importance_mean"],
        color=BLUE,
        edgecolor=CHARCOAL,
        linewidth=0.6,
    )
    axes[0].set_title("Stage-1 context permutation importance", loc="left", fontsize=11)
    axes[0].set_xlabel("Decrease in micro average precision")
    axes[0].grid(axis="x", color=LIGHT_GREY, linewidth=0.8)

    axes[1].barh(
        oracle["feature"],
        oracle["importance_mean"],
        color=GOLD,
        edgecolor=CHARCOAL,
        linewidth=0.6,
    )
    axes[1].set_title("Oracle TabNet symptom-mask importance", loc="left", fontsize=11)
    axes[1].set_xlabel("Mean normalized mask importance")
    axes[1].grid(axis="x", color=LIGHT_GREY, linewidth=0.8)

    for ax in axes:
        ax.set_axisbelow(True)
        sns.despine(ax=ax)
    fig.suptitle(
        "Feature attribution in the revised cascade",
        x=0.07,
        y=0.99,
        ha="left",
        fontsize=14,
    )
    fig.text(
        0.07,
        0.92,
        "Permutation scores use a development fold; TabNet masks are averaged across folds.",
        color=GREY,
        fontsize=9,
    )
    fig.subplots_adjust(top=0.78, wspace=0.48)
    save_figure(fig, "Fig_Publication_Feature_Attribution")


def main() -> None:
    endpoint = pd.read_csv(RESULT_DIR / "endpoint_metrics.csv")
    stage1 = pd.read_csv(RESULT_DIR / "stage1_metrics.csv")
    endpoint_accuracy_figure(endpoint)
    stage1_figure(stage1)
    error_propagation_figure(endpoint)
    confusion_figure()
    calibration_figure()
    importance_figure()
    print(f"Publication figures written to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
