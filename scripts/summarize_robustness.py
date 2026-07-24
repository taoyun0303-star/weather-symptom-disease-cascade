from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "tmp" / "matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


RESULT_DIR = PROJECT_ROOT / "results" / "robustness"
FIGURE_DIR = PROJECT_ROOT / "figures" / "robustness"
METRICS_PATH = RESULT_DIR / "endpoint_metrics.csv"
SUMMARY_PATH = RESULT_DIR / "summary_metrics.csv"

ENDPOINT_ORDER = ["class_prior", "direct_xgb", "cascade_xgb_tabnet"]
ENDPOINT_LABELS = {
    "class_prior": "Class-prior baseline",
    "direct_xgb": "Direct XGBoost",
    "cascade_xgb_tabnet": "Cascade XGBoost -> TabNet",
}
ENDPOINT_COLORS = {
    "class_prior": "#89939B",
    "direct_xgb": "#2F5D8C",
    "cascade_xgb_tabnet": "#B7657B",
}
REQUIRED_COLUMNS = {
    "panel",
    "split_seed",
    "outer_fold",
    "endpoint",
    "status",
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "negative_log_likelihood",
    "ece_10bin",
}


def load_metrics(path: Path = METRICS_PATH) -> pd.DataFrame:
    metrics = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(metrics.columns)
    if missing:
        raise ValueError(f"Missing required metric columns: {sorted(missing)}")
    if metrics.empty:
        raise ValueError("Robustness metrics are empty.")
    if (metrics["status"] != "ok").any():
        failed = metrics.loc[metrics["status"] != "ok", ["panel", "outer_fold", "endpoint"]]
        raise ValueError(f"Cannot summarize failed runs:\n{failed.to_string(index=False)}")
    if set(metrics["endpoint"]) != set(ENDPOINT_ORDER):
        raise ValueError("Unexpected endpoint set in robustness metrics.")
    return metrics


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics.groupby(["panel", "endpoint"], as_index=False)
        .agg(
            runs=("endpoint", "size"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_min=("accuracy", "min"),
            accuracy_max=("accuracy", "max"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            f1_macro_mean=("f1_macro", "mean"),
            nll_mean=("negative_log_likelihood", "mean"),
            ece_mean=("ece_10bin", "mean"),
        )
        .copy()
    )
    summary["endpoint"] = pd.Categorical(
        summary["endpoint"], categories=ENDPOINT_ORDER, ordered=True
    )
    return summary.sort_values(["panel", "endpoint"]).reset_index(drop=True)


def plot_per_run_metrics(metrics: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    panels = [
        ("primary_five_fold", "Five-fold primary", "outer_fold"),
        ("seed_sensitivity", "Seed sensitivity", "split_seed"),
    ]
    metric_specs = [("accuracy", "Accuracy"), ("f1_macro", "Macro F1")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey="col")

    for row, (panel, title, x_column) in enumerate(panels):
        panel_data = metrics.loc[metrics["panel"] == panel].copy()
        for col, (metric, metric_label) in enumerate(metric_specs):
            ax = axes[row, col]
            for endpoint in ENDPOINT_ORDER:
                series = panel_data.loc[panel_data["endpoint"] == endpoint].sort_values(x_column)
                ax.plot(
                    series[x_column].astype(str),
                    100.0 * series[metric],
                    marker="o",
                    linewidth=1.8,
                    markersize=5,
                    label=ENDPOINT_LABELS[endpoint],
                    color=ENDPOINT_COLORS[endpoint],
                )
            ax.set_title(f"{title}: {metric_label}", loc="left", fontsize=11, pad=10)
            ax.set_xlabel("Outer fold" if x_column == "outer_fold" else "Split/model seed")
            ax.set_ylabel(f"{metric_label} (%)")
            ax.grid(axis="y", color="#E8ECEF", linewidth=0.8)
            sns.despine(ax=ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Robustness comparison: direct weather prediction versus the symptom cascade",
        x=0.06,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.93,
        "All points are held-out evaluations; no pooled confidence interval is implied.",
        fontsize=9,
        color="#59636B",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.9))
    fig.savefig(FIGURE_DIR / "Fig_Robustness_Comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "Fig_Robustness_Comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metrics = load_metrics()
    summary = summarize_metrics(metrics)
    summary.to_csv(SUMMARY_PATH, index=False)
    plot_per_run_metrics(metrics)
    print(f"Wrote {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {FIGURE_DIR.relative_to(PROJECT_ROOT) / 'Fig_Robustness_Comparison.png'}")


if __name__ == "__main__":
    main()
