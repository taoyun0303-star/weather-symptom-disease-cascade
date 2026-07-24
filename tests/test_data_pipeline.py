from pathlib import Path

import numpy as np

from weather_health.constants import CORE_SYMPTOMS, FULL_WEATHER_COLS, MODELED_SYMPTOMS
from weather_health.data import compute_weather_indices, load_publication_data
from weather_health.experiment import PublicationExperiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw" / "Weather-related disease prediction.csv"
MD5 = "4aa51b2bb76b45b2000ce71517a0fd1e"


def test_raw_checksum_and_primary_profile():
    frame, profile = load_publication_data(RAW, MD5, remove_exact_duplicates=True)
    assert profile["source_rows"] == 5200
    assert profile["analysis_rows"] == 4981
    assert profile["exact_duplicates_removed"] == 219
    assert len(CORE_SYMPTOMS) == 37
    assert len(MODELED_SYMPTOMS) == 36
    assert frame["shivering"].sum() == 0
    assert set(FULL_WEATHER_COLS).issubset(frame.columns)


def test_humidity_is_fraction_and_indices_are_finite():
    result = compute_weather_indices(
        np.array([30.0, 10.0]),
        np.array([0.70, 0.50]),
        np.array([10.0, 5.0]),
    )
    assert result.shape == (2, 4)
    assert np.isfinite(result.to_numpy()).all()


def test_rejects_percentage_humidity_at_api_boundary():
    try:
        compute_weather_indices(
            np.array([30.0]), np.array([70.0]), np.array([10.0])
        )
    except ValueError as exc:
        assert "fraction" in str(exc)
    else:
        raise AssertionError("Percentage humidity must be rejected at the API boundary.")


def test_experiment_paths_are_independent_of_the_calling_directory():
    experiment = PublicationExperiment(PROJECT_ROOT / "configs" / "publication.yaml")
    assert experiment.project_root == PROJECT_ROOT
    assert experiment.result_dir == PROJECT_ROOT / "results" / "publication"
    assert experiment.processed_dir == PROJECT_ROOT / "data" / "processed"
