from pathlib import Path

import pandas as pd

from weather_health.data import (
    load_publication_data,
    make_publication_split,
    weather_signature,
)
from weather_health.experiment import PublicationExperiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw" / "Weather-related disease prediction.csv"
MD5 = "4aa51b2bb76b45b2000ce71517a0fd1e"


def test_locked_test_is_group_disjoint_and_complete():
    frame, _ = load_publication_data(RAW, MD5, remove_exact_duplicates=True)
    split = make_publication_split(frame, seed=2026, n_splits=5)
    groups = weather_signature(frame)
    assert not (
        set(groups[split.development_indices]) & set(groups[split.test_indices])
    )
    assert len(split.development_indices) + len(split.test_indices) == len(frame)
    assert set(split.development_indices).isdisjoint(set(split.test_indices))


def test_development_folds_are_group_disjoint_and_cover_every_row_once():
    frame, _ = load_publication_data(RAW, MD5, remove_exact_duplicates=True)
    split = make_publication_split(frame, seed=2026, n_splits=5)
    dev_groups = weather_signature(frame)[split.development_indices]
    validation_rows = []
    for train, validation in split.development_folds:
        assert not (set(dev_groups[train]) & set(dev_groups[validation]))
        validation_rows.extend(validation.tolist())
    assert sorted(validation_rows) == list(range(len(split.development_indices)))


def test_split_sensitivity_records_naive_overlap():
    test_output = PROJECT_ROOT / "tmp" / "test_split_sensitivity"
    experiment = PublicationExperiment(
        PROJECT_ROOT / "configs" / "publication.yaml"
    )
    experiment.result_dir = test_output / "results"
    experiment.processed_dir = test_output / "processed"
    experiment.artifact_dir = test_output / "artifacts"
    for directory in (
        experiment.result_dir,
        experiment.processed_dir,
        experiment.artifact_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    experiment._load_data()
    experiment._save_split_sensitivity_profile()

    sensitivity = pd.read_csv(experiment.result_dir / "split_sensitivity.csv")
    indexed = sensitivity.set_index("split")
    assert indexed.loc[
        "weather_signature_group_split", "weather_signatures_shared"
    ] == 0
    assert indexed.loc[
        "naive_stratified_row_split", "weather_signatures_shared"
    ] > 0
