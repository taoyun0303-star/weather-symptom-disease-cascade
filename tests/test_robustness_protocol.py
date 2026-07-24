from weather_health.robustness import build_robustness_specs, core_disease_metrics

import numpy as np


def test_robustness_specs_follow_the_prespecified_panels():
    specs = build_robustness_specs(2026, 5, [2027, 2028, 2029])

    assert [(spec.panel, spec.split_seed, spec.outer_fold) for spec in specs] == [
        ("primary_five_fold", 2026, 0),
        ("primary_five_fold", 2026, 1),
        ("primary_five_fold", 2026, 2),
        ("primary_five_fold", 2026, 3),
        ("primary_five_fold", 2026, 4),
        ("seed_sensitivity", 2027, 0),
        ("seed_sensitivity", 2028, 0),
        ("seed_sensitivity", 2029, 0),
    ]


def test_core_metrics_support_an_outer_fold_with_missing_classes():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
        ]
    )

    metrics = core_disease_metrics(y_true, probabilities, n_classes=3)

    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] < 1.0
    assert metrics["negative_log_likelihood"] >= 0.0
