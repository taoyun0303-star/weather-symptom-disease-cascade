from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_synthetic_concept_study.py"
SPEC = importlib.util.spec_from_file_location("synthetic_concept_study", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
STUDY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STUDY
SPEC.loader.exec_module(STUDY)


def test_generated_dataset_has_grouped_inputs_and_valid_labels() -> None:
    scenario = STUDY.Scenario("test", 0.35, 0.0)
    features, concepts, outcome, groups = STUDY.make_dataset(
        scenario, seed=7, n_groups=12, rows_per_group=3
    )

    assert features.shape == (36, 15)
    assert concepts.shape == (36, 6)
    assert outcome.shape == (36,)
    assert len(np.unique(groups)) == 12
    assert set(np.unique(concepts)).issubset({0, 1})
    assert set(np.unique(outcome)).issubset({0, 1, 2, 3})


def test_summary_reports_direct_minus_cascade_gap() -> None:
    rows = [
        {"scenario": "a", "endpoint": "direct_logistic", "accuracy": 0.7, "f1_macro": 0.6, "concept_f1_macro": np.nan},
        {"scenario": "a", "endpoint": "cascade_logistic", "accuracy": 0.5, "f1_macro": 0.4, "concept_f1_macro": 0.8},
        {"scenario": "a", "endpoint": "class_prior", "accuracy": 0.3, "f1_macro": 0.1, "concept_f1_macro": np.nan},
    ]
    summary = STUDY.summarize(STUDY.pd.DataFrame(rows))
    direct = summary.loc[summary["endpoint"] == "direct_logistic"].iloc[0]

    assert direct["direct_minus_cascade_accuracy"] == pytest.approx(0.2)
