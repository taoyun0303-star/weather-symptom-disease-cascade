from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import xgboost
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .constants import MODELED_SYMPTOMS
from .data import (
    load_publication_data,
    make_publication_split,
    modeling_arrays,
    symptom_signature,
    weather_signature,
)
from .metrics import (
    bootstrap_interval,
    disease_metrics,
    mcnemar_exact,
    multilabel_metrics,
    select_multilabel_thresholds,
)
from .models import (
    cross_validated_disease,
    cross_validated_stage1,
    disease_estimator,
    stage1_estimator,
)
from .paths import resolve_project_path
from .tabnet import cross_validated_tabnet


class PublicationExperiment:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).resolve()
        self.project_root = self.config_path.parent.parent
        with self.config_path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.seed = int(self.config["project"]["random_seed"])
        self.n_jobs = int(self.config["models"]["n_jobs"])
        self.output_config = self.config["outputs"]
        self.processed_dir = resolve_project_path(
            self.output_config["processed_dir"], self.project_root
        )
        self.artifact_dir = resolve_project_path(
            self.output_config["artifact_dir"], self.project_root
        )
        self.result_dir = resolve_project_path(
            self.output_config["result_dir"], self.project_root
        )
        self.figure_dir = resolve_project_path(
            self.output_config["figure_dir"], self.project_root
        )
        for path in (
            self.processed_dir,
            self.artifact_dir,
            self.result_dir,
            self.figure_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        self.stage1_results: dict[str, dict] = {}
        self.endpoint_probabilities: dict[str, np.ndarray] = {}
        self.endpoint_rows: list[dict] = []
        self.per_class_rows: list[pd.DataFrame] = []
        self.confusion_rows: list[pd.DataFrame] = []

    def run(self) -> None:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self._load_data()
        self._save_split_sensitivity_profile()
        self._save_run_manifest()
        self._run_stage1()
        self._run_direct_baselines()
        self._run_primary_cascades()
        self._run_secondary_stage2_controls()
        self._run_context_extensions()
        self._save_pairwise_tests()
        self._save_results()

    def _load_data(self) -> None:
        data_cfg = self.config["data"]
        self.df, self.data_profile = load_publication_data(
            raw_csv=resolve_project_path(data_cfg["raw_csv"], self.project_root),
            expected_md5=data_cfg["expected_md5"],
            remove_exact_duplicates=bool(data_cfg["remove_exact_duplicates"]),
        )
        self.arrays = modeling_arrays(self.df)
        self.split = make_publication_split(
            self.df,
            seed=self.seed,
            n_splits=int(self.config["split"]["development_folds"]),
        )
        self.dev_idx = self.split.development_indices
        self.test_idx = self.split.test_indices
        self.dev_folds = self.split.development_folds

        self.label_encoder = LabelEncoder()
        self.y_all = self.label_encoder.fit_transform(self.arrays["disease"])
        self.y_dev = self.y_all[self.dev_idx]
        self.y_test = self.y_all[self.test_idx]
        self.class_names = self.label_encoder.classes_
        self.n_classes = len(self.class_names)
        self.y_symptoms_dev = self.arrays["symptoms"][self.dev_idx]
        self.y_symptoms_test = self.arrays["symptoms"][self.test_idx]

        self.data_profile.update(
            {
                "development_rows": int(len(self.dev_idx)),
                "locked_test_rows": int(len(self.test_idx)),
                "weather_groups_total": int(
                    pd.Series(weather_signature(self.df)).nunique()
                ),
                "symptom_signatures_total": int(
                    pd.Series(symptom_signature(self.df)).nunique()
                ),
                "weather_group_overlap_development_test": 0,
                "symptom_signature_overlap_test_rows": int(
                    pd.Series(symptom_signature(self.df)[self.test_idx]).isin(
                        set(symptom_signature(self.df)[self.dev_idx])
                    ).sum()
                ),
            }
        )
        self.df.to_csv(
            self.processed_dir / "publication_modeling_data.csv",
            index=False,
            encoding="utf-8",
        )
        with (self.result_dir / "data_quality_profile.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(self.data_profile, handle, ensure_ascii=False, indent=2)
        np.savez_compressed(
            self.artifact_dir / "publication_split.npz",
            development_indices=self.dev_idx,
            test_indices=self.test_idx,
            groups=self.split.groups,
            **{
                f"fold_{i}_train": train
                for i, (train, _) in enumerate(self.dev_folds)
            },
            **{
                f"fold_{i}_validation": validation
                for i, (_, validation) in enumerate(self.dev_folds)
            },
        )
        joblib.dump(
            self.label_encoder, self.artifact_dir / "disease_label_encoder.joblib"
        )

    def _save_split_sensitivity_profile(self) -> None:
        """Compare signature overlap under grouped and naive row splits."""

        if not self.config["experiments"].get(
            "include_group_split_sensitivity", False
        ):
            return

        all_indices = np.arange(len(self.df))
        row_dev, row_test = train_test_split(
            all_indices,
            test_size=len(self.test_idx),
            random_state=self.seed,
            stratify=self.y_all,
        )
        weather = weather_signature(self.df)
        symptoms = symptom_signature(self.df)

        rows = []
        for name, development, test in (
            ("weather_signature_group_split", self.dev_idx, self.test_idx),
            ("naive_stratified_row_split", row_dev, row_test),
        ):
            development_weather = set(weather[development])
            development_symptoms = set(symptoms[development])
            rows.append(
                {
                    "split": name,
                    "development_rows": int(len(development)),
                    "test_rows": int(len(test)),
                    "weather_signatures_shared": int(
                        len(development_weather & set(weather[test]))
                    ),
                    "test_rows_with_seen_weather_signature": int(
                        pd.Series(weather[test]).isin(development_weather).sum()
                    ),
                    "test_rows_with_seen_symptom_signature": int(
                        pd.Series(symptoms[test]).isin(development_symptoms).sum()
                    ),
                }
            )

        pd.DataFrame(rows).to_csv(
            self.result_dir / "split_sensitivity.csv",
            index=False,
            encoding="utf-8",
        )

    def _save_run_manifest(self) -> None:
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_version": self.config["project"]["protocol_version"],
            "config_path": str(self.config_path),
            "config": self.config,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "xgboost": xgboost.__version__,
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            },
            "data_profile": self.data_profile,
        }
        with (self.artifact_dir / "run_manifest.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)

    def _run_stage1_for_features(
        self, name: str, feature_key: str, result_name: str
    ) -> dict:
        x_all = self.arrays[feature_key]
        x_dev, x_test = x_all[self.dev_idx], x_all[self.test_idx]
        estimator = stage1_estimator(name, self.seed, self.n_jobs)
        predictions = cross_validated_stage1(
            estimator,
            x_dev,
            self.y_symptoms_dev,
            x_test,
            self.dev_folds,
        )
        thresholds = select_multilabel_thresholds(
            self.y_symptoms_dev, predictions.development
        )
        dev_summary, dev_per_symptom = multilabel_metrics(
            self.y_symptoms_dev,
            predictions.development,
            thresholds,
            MODELED_SYMPTOMS,
        )
        test_summary, test_per_symptom = multilabel_metrics(
            self.y_symptoms_test,
            predictions.test,
            thresholds,
            MODELED_SYMPTOMS,
        )
        rows = []
        for scope, summary in (
            ("development_oof", dev_summary),
            ("locked_test", test_summary),
        ):
            rows.append(
                {
                    "model": result_name,
                    "feature_set": feature_key,
                    "scope": scope,
                    "fold_logloss_mean": float(np.mean(predictions.fold_losses)),
                    "fold_logloss_std": float(np.std(predictions.fold_losses, ddof=1)),
                    **summary,
                }
            )
        self.stage1_metric_rows.extend(rows)
        for scope, table in (
            ("development_oof", dev_per_symptom),
            ("locked_test", test_per_symptom),
        ):
            enriched = table.copy()
            enriched.insert(0, "scope", scope)
            enriched.insert(0, "model", result_name)
            self.stage1_per_symptom_rows.append(enriched)

        np.savez_compressed(
            self.artifact_dir / f"stage1_{result_name}_probabilities.npz",
            development=predictions.development,
            test=predictions.test,
            thresholds=thresholds,
        )
        return {
            "development": predictions.development,
            "test": predictions.test,
            "thresholds": thresholds,
        }

    def _run_stage1(self) -> None:
        self.stage1_metric_rows: list[dict] = []
        self.stage1_per_symptom_rows: list[pd.DataFrame] = []
        for name in self.config["models"]["stage1"]:
            self.stage1_results[name] = self._run_stage1_for_features(
                name, "weather_full", name
            )

        if self.config["experiments"]["include_engineered_weather_ablation"]:
            self.stage1_results["xgb_raw_weather"] = self._run_stage1_for_features(
                "xgb", "weather_raw", "xgb_raw_weather"
            )

    def _record_endpoint(
        self,
        pipeline: str,
        category: str,
        representation: str,
        stage1: str,
        stage2: str,
        probabilities: np.ndarray,
        development_probabilities: np.ndarray | None = None,
        feature_importance: np.ndarray | None = None,
    ) -> None:
        summary, per_class, matrix = disease_metrics(
            self.y_test, probabilities, self.class_names
        )
        bootstrap_cfg = self.config["split"]
        acc_low, acc_high = bootstrap_interval(
            self.y_test,
            probabilities,
            metric="accuracy",
            replicates=int(bootstrap_cfg["bootstrap_replicates"]),
            seed=int(bootstrap_cfg["bootstrap_seed"]),
        )
        f1_low, f1_high = bootstrap_interval(
            self.y_test,
            probabilities,
            metric="f1_macro",
            replicates=int(bootstrap_cfg["bootstrap_replicates"]),
            seed=int(bootstrap_cfg["bootstrap_seed"]) + 1,
        )
        row = {
            "pipeline": pipeline,
            "category": category,
            "representation": representation,
            "stage1": stage1,
            "stage2": stage2,
            **summary,
            "accuracy_ci_low": acc_low,
            "accuracy_ci_high": acc_high,
            "f1_macro_ci_low": f1_low,
            "f1_macro_ci_high": f1_high,
        }
        self.endpoint_rows.append(row)
        per_class.insert(0, "pipeline", pipeline)
        self.per_class_rows.append(per_class)
        matrix_table = pd.DataFrame(
            matrix, index=self.class_names, columns=self.class_names
        )
        matrix_long = (
            matrix_table.rename_axis("true_class")
            .reset_index()
            .melt(
                id_vars="true_class",
                var_name="predicted_class",
                value_name="count",
            )
        )
        matrix_long.insert(0, "pipeline", pipeline)
        self.confusion_rows.append(matrix_long)
        self.endpoint_probabilities[pipeline] = probabilities
        payload = {
            "test_probabilities": probabilities,
            "test_predictions": probabilities.argmax(axis=1),
            "test_true": self.y_test,
        }
        if development_probabilities is not None:
            payload["development_probabilities"] = development_probabilities
        if feature_importance is not None:
            payload["feature_importance"] = feature_importance
        np.savez_compressed(
            self.artifact_dir / f"endpoint_{pipeline}.npz", **payload
        )

    def _run_direct_baselines(self) -> None:
        x_dev = self.arrays["weather_full"][self.dev_idx]
        x_test = self.arrays["weather_full"][self.test_idx]
        for name in self.config["models"]["direct"]:
            result = cross_validated_disease(
                disease_estimator(name, self.seed, self.n_jobs, self.n_classes),
                x_dev,
                self.y_dev,
                x_test,
                self.dev_folds,
                self.n_classes,
            )
            self._record_endpoint(
                pipeline=f"direct_{name}",
                category="direct_baseline",
                representation="weather_full",
                stage1="none",
                stage2=name,
                probabilities=result.test,
                development_probabilities=result.development,
            )

    def _run_context_extensions(self) -> None:
        if not self.config["experiments"].get("include_context_extensions", False):
            return

        # Direct context baselines.
        for feature_key, label in (
            ("weather_demographic", "demographic"),
            ("pre_event_context", "history"),
        ):
            x_dev = self.arrays[feature_key][self.dev_idx]
            x_test = self.arrays[feature_key][self.test_idx]
            result = cross_validated_disease(
                disease_estimator("xgb", self.seed, self.n_jobs, self.n_classes),
                x_dev,
                self.y_dev,
                x_test,
                self.dev_folds,
                self.n_classes,
            )
            self._record_endpoint(
                pipeline=f"direct_xgb_{label}",
                category="context_extension",
                representation=feature_key,
                stage1="none",
                stage2="xgb",
                probabilities=result.test,
                development_probabilities=result.development,
            )

        # Predict symptoms from all information available before symptom
        # observation, then test both a pure bottleneck and a skip connection.
        context_stage1 = self._run_stage1_for_features(
            "xgb", "pre_event_context", "xgb_pre_event_context"
        )
        self.stage1_results["xgb_pre_event_context"] = context_stage1

        self._tabnet_endpoint(
            pipeline="cascade_context_xgb_prob_tabnet",
            category="context_extension",
            representation="context_predicted_symptom_probability",
            stage1="xgb_pre_event_context",
            x_dev=context_stage1["development"],
            x_test=context_stage1["test"],
        )

        xgb_stage2 = disease_estimator(
            "xgb", self.seed, self.n_jobs, self.n_classes
        )
        bottleneck = cross_validated_disease(
            xgb_stage2,
            context_stage1["development"],
            self.y_dev,
            context_stage1["test"],
            self.dev_folds,
            self.n_classes,
        )
        self._record_endpoint(
            pipeline="cascade_context_xgb_prob_xgb",
            category="context_extension",
            representation="context_predicted_symptom_probability",
            stage1="xgb_pre_event_context",
            stage2="xgb",
            probabilities=bottleneck.test,
            development_probabilities=bottleneck.development,
        )

        skip_dev = np.column_stack(
            [
                self.arrays["pre_event_context"][self.dev_idx],
                context_stage1["development"],
            ]
        )
        skip_test = np.column_stack(
            [
                self.arrays["pre_event_context"][self.test_idx],
                context_stage1["test"],
            ]
        )
        skip = cross_validated_disease(
            disease_estimator("xgb", self.seed, self.n_jobs, self.n_classes),
            skip_dev,
            self.y_dev,
            skip_test,
            self.dev_folds,
            self.n_classes,
        )
        self._record_endpoint(
            pipeline="skip_context_symptom_xgb",
            category="context_extension",
            representation="pre_event_context_plus_symptom_probability",
            stage1="xgb_pre_event_context",
            stage2="xgb",
            probabilities=skip.test,
            development_probabilities=skip.development,
        )

    def _tabnet_endpoint(
        self,
        pipeline: str,
        category: str,
        representation: str,
        stage1: str,
        x_dev: np.ndarray,
        x_test: np.ndarray,
    ) -> None:
        result = cross_validated_tabnet(
            x_dev,
            self.y_dev,
            x_test,
            self.dev_folds,
            self.n_classes,
            self.config["models"]["tabnet"],
            self.seed,
        )
        self._record_endpoint(
            pipeline=pipeline,
            category=category,
            representation=representation,
            stage1=stage1,
            stage2="tabnet",
            probabilities=result.test,
            development_probabilities=result.development,
            feature_importance=result.feature_importance,
        )

    def _run_primary_cascades(self) -> None:
        for name in self.config["models"]["stage1"]:
            stage1 = self.stage1_results[name]
            self._tabnet_endpoint(
                pipeline=f"cascade_{name}_prob_tabnet",
                category="deployable_cascade",
                representation="symptom_probability",
                stage1=name,
                x_dev=stage1["development"],
                x_test=stage1["test"],
            )

        if self.config["experiments"]["include_binary_xgb_ablation"]:
            stage1 = self.stage1_results["xgb"]
            self._tabnet_endpoint(
                pipeline="cascade_xgb_binary_tabnet",
                category="binarization_ablation",
                representation="thresholded_symptom",
                stage1="xgb",
                x_dev=(
                    stage1["development"] >= stage1["thresholds"]
                ).astype(np.float32),
                x_test=(stage1["test"] >= stage1["thresholds"]).astype(np.float32),
            )

        if self.config["experiments"]["include_oracle_tabnet"]:
            self._tabnet_endpoint(
                pipeline="oracle_true_symptom_tabnet",
                category="oracle_upper_bound",
                representation="true_symptom",
                stage1="oracle",
                x_dev=self.y_symptoms_dev.astype(np.float32),
                x_test=self.y_symptoms_test.astype(np.float32),
            )

    def _run_secondary_stage2_controls(self) -> None:
        stage1 = self.stage1_results["xgb"]
        for name in self.config["models"]["stage2_secondary"]:
            result = cross_validated_disease(
                disease_estimator(name, self.seed, self.n_jobs, self.n_classes),
                stage1["development"],
                self.y_dev,
                stage1["test"],
                self.dev_folds,
                self.n_classes,
            )
            self._record_endpoint(
                pipeline=f"cascade_xgb_prob_{name}",
                category="stage2_architecture_control",
                representation="symptom_probability",
                stage1="xgb",
                stage2=name,
                probabilities=result.test,
                development_probabilities=result.development,
            )

    def _save_pairwise_tests(self) -> None:
        primary = "cascade_xgb_prob_tabnet"
        comparisons = [
            "direct_xgb",
            "cascade_xgb_binary_tabnet",
            "oracle_true_symptom_tabnet",
        ]
        rows = []
        primary_predictions = self.endpoint_probabilities[primary].argmax(axis=1)
        for comparison in comparisons:
            if comparison not in self.endpoint_probabilities:
                continue
            result = mcnemar_exact(
                self.y_test,
                primary_predictions,
                self.endpoint_probabilities[comparison].argmax(axis=1),
            )
            rows.append(
                {
                    "pipeline_a": primary,
                    "pipeline_b": comparison,
                    **result,
                }
            )
        pd.DataFrame(rows).to_csv(
            self.result_dir / "pairwise_mcnemar.csv",
            index=False,
            encoding="utf-8",
        )

    def _save_results(self) -> None:
        pd.DataFrame(self.stage1_metric_rows).to_csv(
            self.result_dir / "stage1_metrics.csv", index=False, encoding="utf-8"
        )
        pd.concat(self.stage1_per_symptom_rows, ignore_index=True).to_csv(
            self.result_dir / "stage1_per_symptom.csv",
            index=False,
            encoding="utf-8",
        )
        endpoint_table = pd.DataFrame(self.endpoint_rows).sort_values(
            ["category", "f1_macro"], ascending=[True, False]
        )
        endpoint_table.to_csv(
            self.result_dir / "endpoint_metrics.csv", index=False, encoding="utf-8"
        )
        pd.concat(self.per_class_rows, ignore_index=True).to_csv(
            self.result_dir / "endpoint_per_class.csv",
            index=False,
            encoding="utf-8",
        )
        pd.concat(self.confusion_rows, ignore_index=True).to_csv(
            self.result_dir / "confusion_matrices.csv",
            index=False,
            encoding="utf-8",
        )
        pd.DataFrame(
            {
                "test_row_index": self.test_idx,
                "true_label_encoded": self.y_test,
                "true_label": self.class_names[self.y_test],
            }
        ).to_csv(
            self.result_dir / "locked_test_rows.csv", index=False, encoding="utf-8"
        )
