from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "summarize_robustness.py"
SPEC = importlib.util.spec_from_file_location("summarize_robustness", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def test_summary_aggregates_panel_endpoint_runs() -> None:
    metrics = pd.DataFrame(
        {
            "panel": ["primary", "primary", "primary", "primary", "primary", "primary"],
            "endpoint": [
                "class_prior",
                "class_prior",
                "direct_xgb",
                "direct_xgb",
                "cascade_xgb_tabnet",
                "cascade_xgb_tabnet",
            ],
            "accuracy": [0.2, 0.18, 0.24, 0.22, 0.12, 0.14],
            "balanced_accuracy": [0.09, 0.09, 0.13, 0.14, 0.1, 0.1],
            "f1_macro": [0.03, 0.03, 0.11, 0.12, 0.04, 0.05],
            "negative_log_likelihood": [2.27, 2.28, 2.25, 2.24, 2.39, 2.38],
            "ece_10bin": [0.001, 0.001, 0.08, 0.09, 0.02, 0.03],
        }
    )

    summary = REPORT.summarize_metrics(metrics)
    direct = summary.loc[summary["endpoint"] == "direct_xgb"].iloc[0]

    assert direct["runs"] == 2
    assert direct["accuracy_mean"] == pytest.approx(0.23)
    assert direct["accuracy_min"] == pytest.approx(0.22)
    assert direct["accuracy_max"] == pytest.approx(0.24)
    assert direct["f1_macro_mean"] == pytest.approx(0.115)
