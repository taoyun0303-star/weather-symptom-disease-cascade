from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT_PATH = ROOT / "results" / "publication" / "endpoint_metrics.csv"
PROJECT_DOCS = [
    ROOT / "README.md",
    ROOT / "VALIDATION_REPORT.md",
]


def test_primary_endpoint_values_are_synchronized_with_project_docs() -> None:
    endpoint = pd.read_csv(ENDPOINT_PATH).set_index("pipeline")
    expected_percentages = {
        "cascade_xgb_prob_tabnet": 11.94,
        "cascade_context_xgb_prob_xgb": 39.62,
        "direct_xgb_history": 41.12,
        "oracle_true_symptom_tabnet": 96.29,
    }

    for document_path in PROJECT_DOCS:
        text = document_path.read_text(encoding="utf-8")
        for pipeline, displayed_value in expected_percentages.items():
            observed_value = 100.0 * endpoint.loc[pipeline, "accuracy"]
            assert round(observed_value, 2) == displayed_value
            assert f"{displayed_value:.2f}" in text


def test_project_docs_exclude_superseded_primary_claims() -> None:
    for document_path in PROJECT_DOCS:
        text = document_path.read_text(encoding="utf-8")
        assert "50.63" not in text
        assert "88.44" not in text
