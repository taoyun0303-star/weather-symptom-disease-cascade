from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.preprocessing import LabelEncoder

from .data import load_publication_data, make_publication_split, modeling_arrays
from .metrics import bootstrap_interval, expected_calibration_error
from .models import (
    cross_validated_disease,
    cross_validated_stage1,
    disease_estimator,
    stage1_estimator,
)
from .paths import resolve_project_path
from .tabnet import cross_validated_tabnet


@dataclass(frozen=True)
class RobustnessSpec:
    panel: str
    split_seed: int
    outer_fold: int

    @property
    def model_seed(self) -> int:
        return self.split_seed + self.outer_fold


def build_robustness_specs(
    primary_seed: int,
    n_splits: int,
    sensitivity_seeds: list[int],
) -> list[RobustnessSpec]:
    """Return the fixed resampling panels specified in the protocol."""

    primary = [
        RobustnessSpec("primary_five_fold", primary_seed, fold)
        for fold in range(n_splits)
    ]
    sensitivity = [
        RobustnessSpec("seed_sensitivity", seed, 0)
        for seed in sensitivity_seeds
    ]
    return primary + sensitivity


def core_disease_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, n_classes: int
) -> dict[str, float]:
    """Metrics that remain defined when an outer fold lacks a class."""

    predictions = probabilities.argmax(axis=1)
    labels = np.arange(n_classes)
    one_hot = np.eye(n_classes, dtype=float)[y_true]
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "f1_macro": float(
            f1_score(y_true, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "negative_log_likelihood": float(
            log_loss(y_true, probabilities, labels=labels)
        ),
        "multiclass_brier": float(
            np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
        ),
        "ece_10bin": float(expected_calibration_error(y_true, probabilities)),
    }


class RobustnessExperiment:
    """Run the prespecified group-disjoint robustness extension."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).resolve()
        self.project_root = self.config_path.parent.parent
        with self.config_path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)

        output_config = self.config["outputs"]
        self.artifact_dir = resolve_project_path(
            output_config["artifact_dir"], self.project_root
        )
        self.result_dir = resolve_project_path(
            output_config["result_dir"], self.project_root
        )
        for directory in (self.artifact_dir, self.result_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def run(self, dry_run: bool = False) -> pd.DataFrame:
        data_config = self.config["data"]
        frame, data_profile = load_publication_data(
            resolve_project_path(data_config["raw_csv"], self.project_root),
            data_config["expected_md5"],
            remove_exact_duplicates=bool(data_config["remove_exact_duplicates"]),
        )
        arrays = modeling_arrays(frame)
        label_encoder = LabelEncoder()
        y_all = label_encoder.fit_transform(arrays["disease"])
        n_classes = len(label_encoder.classes_)
        split_config = self.config["split"]
        robustness_config = self.config["robustness"]
        specs = build_robustness_specs(
            primary_seed=int(robustness_config["primary_seed"]),
            n_splits=int(split_config["development_folds"]),
            sensitivity_seeds=[int(seed) for seed in robustness_config["sensitivity_seeds"]],
        )

        split_rows = []
        result_rows = []
        failure_rows = []
        for spec in specs:
            split = make_publication_split(
                frame,
                seed=spec.split_seed,
                n_splits=int(split_config["development_folds"]),
                outer_fold=spec.outer_fold,
            )
            development = split.development_indices
            test = split.test_indices
            test_class_counts = np.bincount(y_all[test], minlength=n_classes)
            split_rows.append(
                {
                    **asdict(spec),
                    "development_rows": int(len(development)),
                    "test_rows": int(len(test)),
                    "development_group_count": int(len(set(split.groups[development]))),
                    "test_group_count": int(len(set(split.groups[test]))),
                    "test_classes_observed": int(np.count_nonzero(test_class_counts)),
                    "test_class_counts": json.dumps(test_class_counts.tolist()),
                }
            )
            if dry_run:
                continue
            self._run_spec(
                spec,
                arrays,
                y_all,
                n_classes,
                split,
                result_rows,
                failure_rows,
            )

        pd.DataFrame(split_rows).to_csv(
            self.result_dir / "split_manifest.csv", index=False, encoding="utf-8"
        )
        pd.DataFrame(result_rows).to_csv(
            self.result_dir / "endpoint_metrics.csv", index=False, encoding="utf-8"
        )
        pd.DataFrame(failure_rows).to_csv(
            self.result_dir / "failures.csv", index=False, encoding="utf-8"
        )
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dry_run": dry_run,
            "config_path": str(self.config_path),
            "config": self.config,
            "data_profile": data_profile,
            "specifications": [asdict(spec) for spec in specs],
        }
        with (self.artifact_dir / "run_manifest.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        return pd.DataFrame(result_rows)

    def _run_spec(
        self,
        spec: RobustnessSpec,
        arrays: dict[str, np.ndarray],
        y_all: np.ndarray,
        n_classes: int,
        split,
        result_rows: list[dict],
        failure_rows: list[dict],
    ) -> None:
        development = split.development_indices
        test = split.test_indices
        x_weather = arrays["weather_full"]
        y_symptoms = arrays["symptoms"]
        y_development = y_all[development]
        y_test = y_all[test]
        n_jobs = int(self.config["models"]["n_jobs"])
        tabnet_parameters = self.config["models"]["tabnet"]
        bootstrap_replicates = int(self.config["split"]["bootstrap_replicates"])
        bootstrap_seed = int(self.config["split"]["bootstrap_seed"])

        endpoint_probabilities: dict[str, np.ndarray] = {}
        try:
            endpoint_probabilities["class_prior"] = cross_validated_disease(
                disease_estimator("dummy", spec.model_seed, n_jobs, n_classes),
                x_weather[development],
                y_development,
                x_weather[test],
                split.development_folds,
                n_classes,
            ).test
            endpoint_probabilities["direct_xgb"] = cross_validated_disease(
                disease_estimator("xgb", spec.model_seed, n_jobs, n_classes),
                x_weather[development],
                y_development,
                x_weather[test],
                split.development_folds,
                n_classes,
            ).test
        except Exception as exc:  # record infeasible folds rather than retrying them
            failure_rows.append({**asdict(spec), "endpoint": "direct_models", "error": repr(exc)})
            return

        try:
            stage1 = cross_validated_stage1(
                stage1_estimator("xgb", spec.model_seed, n_jobs),
                x_weather[development],
                y_symptoms[development],
                x_weather[test],
                split.development_folds,
            )
            endpoint_probabilities["cascade_xgb_tabnet"] = cross_validated_tabnet(
                stage1.development,
                y_development,
                stage1.test,
                split.development_folds,
                n_classes,
                tabnet_parameters,
                spec.model_seed,
            ).test
        except Exception as exc:  # preserve direct endpoints if the cascade is infeasible
            failure_rows.append({**asdict(spec), "endpoint": "cascade_xgb_tabnet", "error": repr(exc)})

        for endpoint, probabilities in endpoint_probabilities.items():
            metrics = core_disease_metrics(y_test, probabilities, n_classes)
            accuracy_ci = bootstrap_interval(
                y_test,
                probabilities,
                metric="accuracy",
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + spec.model_seed,
            )
            macro_f1_ci = bootstrap_interval(
                y_test,
                probabilities,
                metric="f1_macro",
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + spec.model_seed + 1,
            )
            result_rows.append(
                {
                    **asdict(spec),
                    "endpoint": endpoint,
                    "status": "ok",
                    **metrics,
                    "accuracy_ci_low": accuracy_ci[0],
                    "accuracy_ci_high": accuracy_ci[1],
                    "f1_macro_ci_low": macro_f1_ci[0],
                    "f1_macro_ci_high": macro_f1_ci[1],
                }
            )
