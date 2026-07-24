from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import log_loss


@dataclass
class TabNetProbabilities:
    development: np.ndarray
    test: np.ndarray
    fold_losses: list[float]
    feature_importance: np.ndarray


def cross_validated_tabnet(
    x_development: np.ndarray,
    y_development: np.ndarray,
    x_test: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_classes: int,
    parameters: dict,
    seed: int,
) -> TabNetProbabilities:
    x_development = np.asarray(x_development, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    oof = np.zeros((len(y_development), n_classes), dtype=float)
    test_probabilities = []
    fold_losses = []
    fold_importances = []

    for fold_number, (train_indices, validation_indices) in enumerate(folds):
        fold_seed = seed + fold_number
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)

        model = TabNetClassifier(
            n_d=int(parameters["n_d"]),
            n_a=int(parameters["n_a"]),
            n_steps=int(parameters["n_steps"]),
            gamma=1.5,
            lambda_sparse=1e-4,
            optimizer_fn=torch.optim.AdamW,
            optimizer_params={"lr": 2e-3, "weight_decay": 1e-5},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={
                "step_size": 10,
                "gamma": 0.7,
                "is_batch_level": False,
            },
            mask_type="sparsemax",
            seed=fold_seed,
            verbose=0,
            device_name="auto",
        )
        model.fit(
            x_development[train_indices],
            y_development[train_indices],
            eval_set=[
                (
                    x_development[validation_indices],
                    y_development[validation_indices],
                )
            ],
            eval_name=["validation"],
            eval_metric=["logloss"],
            max_epochs=int(parameters["max_epochs"]),
            patience=int(parameters["patience"]),
            batch_size=int(parameters["batch_size"]),
            virtual_batch_size=int(parameters["virtual_batch_size"]),
            num_workers=0,
            drop_last=False,
            weights=1,
        )
        validation_proba = model.predict_proba(
            x_development[validation_indices]
        )
        oof[validation_indices] = validation_proba
        test_probabilities.append(model.predict_proba(x_test))
        fold_losses.append(
            float(
                log_loss(
                    y_development[validation_indices],
                    validation_proba,
                    labels=np.arange(n_classes),
                )
            )
        )
        fold_importances.append(np.asarray(model.feature_importances_))

    return TabNetProbabilities(
        development=oof,
        test=np.mean(test_probabilities, axis=0),
        fold_losses=fold_losses,
        feature_importance=np.mean(fold_importances, axis=0),
    )
