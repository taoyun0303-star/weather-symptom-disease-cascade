from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


@dataclass
class CrossValidatedProbabilities:
    development: np.ndarray
    test: np.ndarray
    fold_losses: list[float]


def stage1_estimator(name: str, seed: int, n_jobs: int):
    if name == "lr":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MultiOutputClassifier(
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        )
    if name == "mlp":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=1e-4,
                        batch_size=128,
                        learning_rate_init=1e-3,
                        max_iter=400,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=20,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "xgb":
        base = XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.9,
            min_child_weight=2,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
        )
        return MultiOutputClassifier(base, n_jobs=n_jobs)
    raise ValueError(f"Unknown stage-1 model: {name}")


def disease_estimator(name: str, seed: int, n_jobs: int, n_classes: int):
    if name == "dummy":
        return DummyClassifier(strategy="prior", random_state=seed)
    if name == "lr":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=n_jobs,
        )
    if name == "mlp":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        alpha=1e-4,
                        batch_size=128,
                        learning_rate_init=1e-3,
                        max_iter=500,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=25,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "xgb":
        return XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.9,
            min_child_weight=2,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=n_jobs,
        )
    raise ValueError(f"Unknown disease model: {name}")


def multilabel_probability(estimator, x: np.ndarray) -> np.ndarray:
    probabilities = estimator.predict_proba(x)
    if isinstance(probabilities, list):
        columns = []
        for values in probabilities:
            if values.shape[1] == 1:
                columns.append(np.zeros(len(values), dtype=float))
            else:
                columns.append(values[:, 1])
        return np.column_stack(columns)
    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 3:
        return probabilities[:, :, 1].T
    return probabilities


def cross_validated_stage1(
    estimator,
    x_development: np.ndarray,
    y_development: np.ndarray,
    x_test: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> CrossValidatedProbabilities:
    oof = np.zeros_like(y_development, dtype=float)
    test_predictions = []
    fold_losses = []
    for train_indices, validation_indices in folds:
        model = clone(estimator)
        model.fit(x_development[train_indices], y_development[train_indices])
        validation_proba = multilabel_probability(
            model, x_development[validation_indices]
        )
        oof[validation_indices] = validation_proba
        test_predictions.append(multilabel_probability(model, x_test))
        losses = []
        for column in range(y_development.shape[1]):
            losses.append(
                log_loss(
                    y_development[validation_indices, column],
                    validation_proba[:, column],
                    labels=[0, 1],
                )
            )
        fold_losses.append(float(np.mean(losses)))
    return CrossValidatedProbabilities(
        development=oof,
        test=np.mean(test_predictions, axis=0),
        fold_losses=fold_losses,
    )


def cross_validated_disease(
    estimator,
    x_development: np.ndarray,
    y_development: np.ndarray,
    x_test: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_classes: int,
) -> CrossValidatedProbabilities:
    oof = np.zeros((len(y_development), n_classes), dtype=float)
    test_predictions = []
    fold_losses = []
    for train_indices, validation_indices in folds:
        model = clone(estimator)
        model.fit(x_development[train_indices], y_development[train_indices])
        validation_proba = model.predict_proba(x_development[validation_indices])
        oof[validation_indices] = validation_proba
        test_predictions.append(model.predict_proba(x_test))
        fold_losses.append(
            float(
                log_loss(
                    y_development[validation_indices],
                    validation_proba,
                    labels=np.arange(n_classes),
                )
            )
        )
    return CrossValidatedProbabilities(
        development=oof,
        test=np.mean(test_predictions, axis=0),
        fold_losses=fold_losses,
    )
