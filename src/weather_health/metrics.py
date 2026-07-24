from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == y_true
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def disease_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    predictions = probabilities.argmax(axis=1)
    labels = np.arange(len(class_names))
    y_binary = label_binarize(y_true, classes=labels)
    one_hot = np.eye(len(class_names), dtype=float)[y_true]

    summary = {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "f1_macro": f1_score(y_true, predictions, average="macro"),
        "f1_weighted": f1_score(y_true, predictions, average="weighted"),
        "roc_auc_ovr_macro": roc_auc_score(
            y_binary, probabilities, average="macro", multi_class="ovr"
        ),
        "average_precision_macro": average_precision_score(
            y_binary, probabilities, average="macro"
        ),
        "negative_log_likelihood": log_loss(
            y_true, probabilities, labels=labels
        ),
        "multiclass_brier": np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)),
        "ece_10bin": expected_calibration_error(y_true, probabilities, n_bins=10),
    }
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, labels=labels, zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "class": class_names,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    matrix = confusion_matrix(y_true, predictions, labels=labels)
    return summary, per_class, matrix


def select_multilabel_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.01)
    thresholds = np.full(y_true.shape[1], 0.5, dtype=float)
    for column in range(y_true.shape[1]):
        scores = [
            f1_score(
                y_true[:, column],
                probabilities[:, column] >= threshold,
                zero_division=0,
            )
            for threshold in grid
        ]
        thresholds[column] = float(grid[int(np.argmax(scores))])
    return thresholds


def multilabel_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    names: list[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    predictions = probabilities >= thresholds
    summary = {
        "f1_micro": f1_score(y_true, predictions, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, predictions, average="macro", zero_division=0),
        "average_precision_micro": average_precision_score(
            y_true, probabilities, average="micro"
        ),
        "average_precision_macro": average_precision_score(
            y_true, probabilities, average="macro"
        ),
        "hamming_loss": hamming_loss(y_true, predictions),
    }
    rows = []
    for index, name in enumerate(names):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true[:, index],
            predictions[:, index],
            average="binary",
            zero_division=0,
        )
        rows.append(
            {
                "symptom": name,
                "prevalence": float(y_true[:, index].mean()),
                "threshold": float(thresholds[index]),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "average_precision": float(
                    average_precision_score(y_true[:, index], probabilities[:, index])
                ),
            }
        )
    return summary, pd.DataFrame(rows)


def bootstrap_interval(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    metric: str,
    replicates: int = 2000,
    seed: int = 2407,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    predictions = probabilities.argmax(axis=1)
    values = []
    for _ in range(replicates):
        sample = rng.integers(0, len(y_true), size=len(y_true))
        if metric == "accuracy":
            value = accuracy_score(y_true[sample], predictions[sample])
        elif metric == "f1_macro":
            value = f1_score(
                y_true[sample], predictions[sample], average="macro", zero_division=0
            )
        else:
            raise ValueError(f"Unsupported bootstrap metric: {metric}")
        values.append(value)
    return tuple(float(v) for v in np.quantile(values, [0.025, 0.975]))


def mcnemar_exact(
    y_true: np.ndarray, predictions_a: np.ndarray, predictions_b: np.ndarray
) -> dict[str, float | int]:
    correct_a = predictions_a == y_true
    correct_b = predictions_b == y_true
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    discordant = a_only + b_only
    p_value = (
        float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "a_correct_b_wrong": a_only,
        "a_wrong_b_correct": b_only,
        "discordant": discordant,
        "p_value_exact": p_value,
    }
