from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from weather_health.constants import (
    MODELED_SYMPTOMS,
    PRE_EVENT_CONTEXT_COLS,
)
from weather_health.data import (
    load_publication_data,
    make_publication_split,
    modeling_arrays,
)
from weather_health.models import (
    disease_estimator,
    multilabel_probability,
    stage1_estimator,
)


def repeated_permutation_drop(
    model,
    x_validation: np.ndarray,
    scorer,
    feature_names: list[str],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    baseline = float(scorer(x_validation))
    rows = []
    for feature_index, feature_name in enumerate(feature_names):
        scores = []
        for _ in range(repeats):
            permuted = x_validation.copy()
            permuted[:, feature_index] = rng.permutation(
                permuted[:, feature_index]
            )
            scores.append(float(scorer(permuted)))
        rows.append(
            {
                "feature": feature_name,
                "baseline_score": baseline,
                "permuted_score_mean": float(np.mean(scores)),
                "importance_mean": float(baseline - np.mean(scores)),
                "importance_std": float(np.std(scores, ddof=1)),
                "repeats": repeats,
            }
        )
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def main() -> None:
    data, _ = load_publication_data(
        PROJECT_ROOT / "data" / "raw" / "Weather-related disease prediction.csv",
        "4aa51b2bb76b45b2000ce71517a0fd1e",
        remove_exact_duplicates=True,
    )
    split = make_publication_split(data, seed=2026, n_splits=5)
    arrays = modeling_arrays(data)
    encoder = LabelEncoder()
    y_disease = encoder.fit_transform(arrays["disease"])
    dev = split.development_indices
    train_fold, validation_fold = split.development_folds[0]

    x_context = arrays["pre_event_context"][dev]
    y_dev = y_disease[dev]
    disease_model = disease_estimator(
        "xgb", seed=2026, n_jobs=1, n_classes=len(encoder.classes_)
    )
    disease_model.fit(x_context[train_fold], y_dev[train_fold])

    def disease_scorer(x):
        return f1_score(
            y_dev[validation_fold],
            disease_model.predict(x),
            average="macro",
            zero_division=0,
        )

    disease_importance = repeated_permutation_drop(
        disease_model,
        x_context[validation_fold],
        disease_scorer,
        PRE_EVENT_CONTEXT_COLS,
        repeats=10,
        seed=2407,
    )
    disease_importance.insert(0, "analysis", "direct_context_xgb_macro_f1")

    y_symptoms = arrays["symptoms"][dev]
    stage1_model = stage1_estimator("xgb", seed=2026, n_jobs=1)
    stage1_model.fit(x_context[train_fold], y_symptoms[train_fold])

    def stage1_scorer(x):
        return average_precision_score(
            y_symptoms[validation_fold],
            multilabel_probability(stage1_model, x),
            average="micro",
        )

    stage1_importance = repeated_permutation_drop(
        stage1_model,
        x_context[validation_fold],
        stage1_scorer,
        PRE_EVENT_CONTEXT_COLS,
        repeats=5,
        seed=2408,
    )
    stage1_importance.insert(0, "analysis", "stage1_context_xgb_micro_ap")

    tabnet_rows = []
    for artifact_name, analysis in (
        ("endpoint_oracle_true_symptom_tabnet.npz", "oracle_tabnet_mask"),
        (
            "endpoint_cascade_context_xgb_prob_tabnet.npz",
            "context_probability_tabnet_mask",
        ),
    ):
        artifact = np.load(PROJECT_ROOT / "artifacts" / "publication" / artifact_name)
        for feature, importance in zip(
            MODELED_SYMPTOMS, artifact["feature_importance"]
        ):
            tabnet_rows.append(
                {
                    "analysis": analysis,
                    "feature": feature,
                    "baseline_score": np.nan,
                    "permuted_score_mean": np.nan,
                    "importance_mean": float(importance),
                    "importance_std": np.nan,
                    "repeats": 5,
                }
            )

    output = pd.concat(
        [
            disease_importance,
            stage1_importance,
            pd.DataFrame(tabnet_rows),
        ],
        ignore_index=True,
    )
    output_path = PROJECT_ROOT / "results" / "publication" / "feature_importance.csv"
    output.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Feature importance written to: {output_path}")


if __name__ == "__main__":
    main()
