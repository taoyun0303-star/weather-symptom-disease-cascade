from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "synthetic"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "synthetic"
N_GROUPS = 480
ROWS_PER_GROUP = 6
MASTER_SEEDS = (2026, 2027, 2028)
OUTER_FOLDS = 5
INNER_FOLDS = 4


@dataclass(frozen=True)
class Scenario:
    name: str
    concept_measurement_noise: float
    direct_residual_strength: float


SCENARIOS = (
    Scenario("fully_mediated", 0.35, 0.00),
    Scenario("concept_noise", 1.50, 0.00),
    Scenario("residual_signal", 0.35, 1.00),
)


def _draw_categorical(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.array([rng.choice(probabilities.shape[1], p=row) for row in probabilities])


def make_dataset(
    scenario: Scenario,
    seed: int,
    n_groups: int = N_GROUPS,
    rows_per_group: int = ROWS_PER_GROUP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate grouped inputs, binary concepts, outcome labels, and groups."""

    rng = np.random.default_rng(seed)
    group_latent = rng.normal(size=(n_groups, 6))
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    latent = group_latent[groups] + rng.normal(scale=0.35, size=(len(groups), 6))
    concepts = rng.binomial(1, expit(latent)).astype(int)

    measured_concepts = latent + rng.normal(
        scale=scenario.concept_measurement_noise, size=latent.shape
    )
    residual_features = rng.normal(size=(len(groups), 3))
    nuisance_features = rng.normal(size=(len(groups), 6))

    concept_weights = np.array(
        [
            [1.20, -0.55, 0.85, -0.80],
            [-0.75, 1.10, -0.45, 0.90],
            [0.90, 0.30, -1.05, 0.60],
            [-0.40, 0.95, 0.70, -0.85],
            [0.65, -0.95, 0.25, 0.75],
            [-0.85, 0.45, 1.05, -0.25],
        ]
    )
    residual_weights = np.array(
        [[1.30, -0.50, 0.75, -0.80], [-0.70, 1.15, -0.60, 0.90], [0.45, 0.70, -1.10, 0.60]]
    )
    logits = concepts @ concept_weights
    logits += scenario.direct_residual_strength * (residual_features @ residual_weights)
    logits += np.array([0.10, -0.05, 0.00, -0.05])
    outcome = _draw_categorical(softmax(logits, axis=1), rng)
    features = np.hstack([measured_concepts, residual_features, nuisance_features])
    return features, concepts, outcome, groups


def _classifier(seed: int) -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=1000, random_state=seed)


def _concept_probabilities(
    x_train: np.ndarray,
    concepts_train: np.ndarray,
    x_predict: np.ndarray,
    seed: int,
) -> np.ndarray:
    probabilities = np.zeros((len(x_predict), concepts_train.shape[1]), dtype=float)
    for concept_index in range(concepts_train.shape[1]):
        model = _classifier(seed + concept_index)
        model.fit(x_train, concepts_train[:, concept_index])
        probabilities[:, concept_index] = model.predict_proba(x_predict)[:, 1]
    return probabilities


def _oof_concept_probabilities(
    x_development: np.ndarray,
    concepts_development: np.ndarray,
    groups_development: np.ndarray,
    seed: int,
) -> np.ndarray:
    probabilities = np.zeros_like(concepts_development, dtype=float)
    splitter = GroupKFold(n_splits=INNER_FOLDS)
    for train_index, validation_index in splitter.split(x_development, groups=groups_development):
        probabilities[validation_index] = _concept_probabilities(
            x_development[train_index],
            concepts_development[train_index],
            x_development[validation_index],
            seed,
        )
    return probabilities


def _metrics(y_true: np.ndarray, y_predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_predicted)),
        "f1_macro": float(f1_score(y_true, y_predicted, average="macro", zero_division=0)),
    }


def run_spec(
    scenario: Scenario,
    master_seed: int,
    outer_fold: int,
) -> list[dict[str, float | int | str]]:
    x, concepts, y, groups = make_dataset(scenario, master_seed)
    splitter = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=master_seed)
    splits = list(splitter.split(x, y, groups))
    development_index, test_index = splits[outer_fold]
    x_development, x_test = x[development_index], x[test_index]
    y_development, y_test = y[development_index], y[test_index]
    concepts_development, concepts_test = concepts[development_index], concepts[test_index]
    groups_development = groups[development_index]
    model_seed = master_seed + outer_fold
    metadata = {
        "scenario": scenario.name,
        "master_seed": master_seed,
        "outer_fold": outer_fold,
        "development_rows": len(development_index),
        "test_rows": len(test_index),
        "development_groups": len(np.unique(groups_development)),
        "test_groups": len(np.unique(groups[test_index])),
    }

    class_prior = np.bincount(y_development, minlength=4).argmax()
    baseline_metrics = _metrics(y_test, np.full(len(y_test), class_prior))

    direct_model = _classifier(model_seed)
    direct_model.fit(x_development, y_development)
    direct_metrics = _metrics(y_test, direct_model.predict(x_test))

    development_concept_probabilities = _oof_concept_probabilities(
        x_development, concepts_development, groups_development, model_seed
    )
    test_concept_probabilities = _concept_probabilities(
        x_development, concepts_development, x_test, model_seed
    )
    cascade_model = _classifier(model_seed)
    cascade_model.fit(development_concept_probabilities, y_development)
    cascade_metrics = _metrics(y_test, cascade_model.predict(test_concept_probabilities))
    concept_predictions = (test_concept_probabilities >= 0.5).astype(int)
    concept_f1 = float(
        f1_score(concepts_test, concept_predictions, average="macro", zero_division=0)
    )

    return [
        {**metadata, "endpoint": "class_prior", **baseline_metrics, "concept_f1_macro": np.nan},
        {**metadata, "endpoint": "direct_logistic", **direct_metrics, "concept_f1_macro": np.nan},
        {**metadata, "endpoint": "cascade_logistic", **cascade_metrics, "concept_f1_macro": concept_f1},
    ]


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    means = (
        results.groupby(["scenario", "endpoint"], as_index=False)
        .agg(
            runs=("endpoint", "size"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_min=("accuracy", "min"),
            accuracy_max=("accuracy", "max"),
            f1_macro_mean=("f1_macro", "mean"),
            concept_f1_macro_mean=("concept_f1_macro", "mean"),
        )
        .sort_values(["scenario", "endpoint"])
    )
    direct = means.loc[means["endpoint"] == "direct_logistic", ["scenario", "accuracy_mean"]]
    cascade = means.loc[means["endpoint"] == "cascade_logistic", ["scenario", "accuracy_mean"]]
    gaps = direct.merge(cascade, on="scenario", suffixes=("_direct", "_cascade"))
    gaps["direct_minus_cascade_accuracy"] = gaps["accuracy_mean_direct"] - gaps["accuracy_mean_cascade"]
    return means.merge(gaps[["scenario", "direct_minus_cascade_accuracy"]], on="scenario", how="left")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    specifications = [
        {**asdict(scenario), "master_seed": seed, "outer_fold": fold}
        for scenario in SCENARIOS
        for seed in MASTER_SEEDS
        for fold in range(OUTER_FOLDS)
    ]
    pd.DataFrame(specifications).to_csv(RESULT_DIR / "specifications.csv", index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": arguments.dry_run,
        "scenarios": [asdict(scenario) for scenario in SCENARIOS],
        "master_seeds": MASTER_SEEDS,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "n_groups": N_GROUPS,
        "rows_per_group": ROWS_PER_GROUP,
    }
    (ARTIFACT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if arguments.dry_run:
        print(f"Prepared {len(specifications)} specifications without fitting models.")
        return

    rows: list[dict[str, float | int | str]] = []
    failures: list[dict[str, str | int]] = []
    for specification in specifications:
        scenario = next(item for item in SCENARIOS if item.name == specification["name"])
        try:
            rows.extend(run_spec(scenario, specification["master_seed"], specification["outer_fold"]))
        except Exception as error:
            failures.append({**specification, "error": repr(error)})
    results = pd.DataFrame(rows)
    results.to_csv(RESULT_DIR / "endpoint_metrics.csv", index=False)
    pd.DataFrame(failures).to_csv(RESULT_DIR / "failures.csv", index=False)
    if failures:
        raise RuntimeError(f"Synthetic study recorded {len(failures)} failed specifications.")
    summarize(results).to_csv(RESULT_DIR / "summary_metrics.csv", index=False)
    print(f"Completed {len(specifications)} specifications and wrote {len(results)} endpoint rows.")


if __name__ == "__main__":
    main()
