from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .constants import (
    CORE_SYMPTOMS,
    ENGINEERED_WEATHER_COLS,
    FULL_WEATHER_COLS,
    PRE_EVENT_CONTEXT_COLS,
    MODELED_SYMPTOMS,
    RAW_WEATHER_COLS,
    WEATHER_DEMOGRAPHIC_COLS,
    SOURCE_DUPLICATE_SYMPTOM,
    TARGET_COL,
)


@dataclass(frozen=True)
class PublicationSplit:
    development_indices: np.ndarray
    test_indices: np.ndarray
    development_folds: list[tuple[np.ndarray, np.ndarray]]
    groups: np.ndarray


def md5sum(path: str | Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _humidity_percent(humidity_fraction: np.ndarray) -> np.ndarray:
    humidity = np.asarray(humidity_fraction, dtype=float)
    if np.nanmin(humidity) < 0 or np.nanmax(humidity) > 1.0 + 1e-9:
        raise ValueError("Humidity must be supplied as a fraction in [0, 1].")
    return humidity * 100.0


def compute_weather_indices(
    temperature_c: np.ndarray,
    humidity_fraction: np.ndarray,
    wind_kmh: np.ndarray,
) -> pd.DataFrame:
    """Compute four weather indices with explicit and internally consistent units."""

    t_c = np.asarray(temperature_c, dtype=float)
    rh = _humidity_percent(np.asarray(humidity_fraction, dtype=float))
    wind_kmh = np.asarray(wind_kmh, dtype=float)
    wind_ms = wind_kmh / 3.6

    # NOAA Rothfusz heat-index regression in Fahrenheit. Outside its usual
    # warm/humid domain, air temperature is retained as the conservative value.
    t_f = t_c * 9.0 / 5.0 + 32.0
    hi_f = (
        -42.379
        + 2.04901523 * t_f
        + 10.14333127 * rh
        - 0.22475541 * t_f * rh
        - 0.00683783 * t_f**2
        - 0.05481717 * rh**2
        + 0.00122874 * t_f**2 * rh
        + 0.00085282 * t_f * rh**2
        - 0.00000199 * t_f**2 * rh**2
    )
    hi_c_regression = (hi_f - 32.0) * 5.0 / 9.0
    heat_index = np.where((t_c >= 26.7) & (rh >= 40.0), hi_c_regression, t_c)

    # Magnus dew-point approximation followed by the standard humidex formula.
    rh_safe = np.clip(rh, 1e-6, 100.0)
    gamma = np.log(rh_safe / 100.0) + (17.625 * t_c) / (243.04 + t_c)
    dew_point_c = 243.04 * gamma / (17.625 - gamma)
    vapor_pressure_hpa = 6.11 * np.exp(
        5417.7530 * (1.0 / 273.16 - 1.0 / (273.15 + dew_point_c))
    )
    humidex = t_c + (5.0 / 9.0) * (vapor_pressure_hpa - 10.0)

    # Australian Bureau of Meteorology apparent-temperature formula.
    vapor_pressure = (rh / 100.0) * 6.105 * np.exp(
        17.27 * t_c / (237.7 + t_c)
    )
    apparent_temperature = t_c + 0.33 * vapor_pressure - 0.70 * wind_ms - 4.0

    # Temperature-humidity index in degrees Celsius.
    thi = t_c - (0.55 - 0.0055 * rh) * (t_c - 14.5)

    return pd.DataFrame(
        {
            "heat_index": heat_index,
            "humidex": humidex,
            "apparent_temperature": apparent_temperature,
            "thi": thi,
        }
    )


def load_publication_data(
    raw_csv: str | Path,
    expected_md5: str,
    remove_exact_duplicates: bool = True,
) -> tuple[pd.DataFrame, dict]:
    raw_csv = Path(raw_csv)
    observed_md5 = md5sum(raw_csv)
    if observed_md5.lower() != expected_md5.lower():
        raise ValueError(
            f"Raw-data checksum mismatch: expected {expected_md5}, got {observed_md5}"
        )

    source = pd.read_csv(raw_csv)
    required = set(RAW_WEATHER_COLS + CORE_SYMPTOMS + [TARGET_COL])
    required.discard("pain_behind_eyes")
    required.add(SOURCE_DUPLICATE_SYMPTOM)
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Missing required source columns: {missing}")

    if source.isna().any().any():
        raise ValueError("The publication pipeline requires explicit missing-data handling.")

    source_rows = len(source)
    exact_duplicates = int(source.duplicated().sum())
    if remove_exact_duplicates:
        source = source.drop_duplicates().reset_index(drop=True)
    else:
        source = source.reset_index(drop=True)

    source["pain_behind_eyes"] = source[
        [SOURCE_DUPLICATE_SYMPTOM, "pain_behind_eyes"]
    ].max(axis=1)

    weather_indices = compute_weather_indices(
        source["Temperature (C)"].to_numpy(),
        source["Humidity"].to_numpy(),
        source["Wind Speed (km/h)"].to_numpy(),
    )
    for column in ENGINEERED_WEATHER_COLS:
        source[column] = weather_indices[column].to_numpy()

    for column in CORE_SYMPTOMS:
        values = set(source[column].unique())
        if not values.issubset({0, 1}):
            raise ValueError(f"Symptom {column} is not binary: {sorted(values)}")

    profile = {
        "source_rows": source_rows,
        "analysis_rows": len(source),
        "source_columns": 51,
        "analysis_columns": len(source.columns),
        "exact_duplicates_removed": exact_duplicates if remove_exact_duplicates else 0,
        "md5": observed_md5,
        "humidity_min": float(source["Humidity"].min()),
        "humidity_max": float(source["Humidity"].max()),
        "class_counts": {
            str(k): int(v) for k, v in source[TARGET_COL].value_counts().items()
        },
    }
    return source, profile


def weather_signature(df: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(
        df[RAW_WEATHER_COLS], index=False
    ).astype(str).to_numpy()


def symptom_signature(df: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(
        df[CORE_SYMPTOMS], index=False
    ).astype(str).to_numpy()


def make_publication_split(
    df: pd.DataFrame,
    seed: int = 2026,
    n_splits: int = 5,
    outer_fold: int = 0,
) -> PublicationSplit:
    y = df[TARGET_COL].to_numpy()
    groups = weather_signature(df)
    all_indices = np.arange(len(df))

    outer = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    outer_folds = list(outer.split(all_indices, y, groups=groups))
    if not 0 <= outer_fold < len(outer_folds):
        raise ValueError(
            f"outer_fold must be in [0, {len(outer_folds) - 1}], got {outer_fold}."
        )
    development_indices, test_indices = outer_folds[outer_fold]

    dev_y = y[development_indices]
    dev_groups = groups[development_indices]
    inner = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed + 1
    )
    development_folds = [
        (train_fold.astype(int), validation_fold.astype(int))
        for train_fold, validation_fold in inner.split(
            np.arange(len(development_indices)), dev_y, groups=dev_groups
        )
    ]

    if set(groups[development_indices]) & set(groups[test_indices]):
        raise AssertionError("Weather groups overlap between development and test.")
    for train_fold, validation_fold in development_folds:
        if set(dev_groups[train_fold]) & set(dev_groups[validation_fold]):
            raise AssertionError("Weather groups overlap inside a development fold.")

    return PublicationSplit(
        development_indices=development_indices.astype(int),
        test_indices=test_indices.astype(int),
        development_folds=development_folds,
        groups=groups,
    )


def modeling_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "weather_raw": df[RAW_WEATHER_COLS].to_numpy(dtype=np.float32),
        "weather_full": df[FULL_WEATHER_COLS].to_numpy(dtype=np.float32),
        "weather_demographic": df[WEATHER_DEMOGRAPHIC_COLS].to_numpy(
            dtype=np.float32
        ),
        "pre_event_context": df[PRE_EVENT_CONTEXT_COLS].to_numpy(dtype=np.float32),
        "symptoms": df[MODELED_SYMPTOMS].to_numpy(dtype=np.int8),
        "disease": df[TARGET_COL].astype(str).to_numpy(),
    }
