# Dataset provenance and modeling boundary

## Source

- Title: Weather-related Disease Prediction Dataset
- Creators: Ali Shan, Iqra Amir, and Mustafa Kamal
- Publisher: Zenodo
- Version: v1, published 2024-05-28
- DOI: https://doi.org/10.5281/zenodo.11366485
- License: Creative Commons Attribution 4.0 International
- Source file: `data/raw/Weather-related disease prediction.csv` (downloaded
  locally with `python scripts/fetch_dataset.py`; not committed by default)
- Official MD5: `4aa51b2bb76b45b2000ce71517a0fd1e`

## Observed data

The source contains 5,200 rows and 51 columns. It includes age, binary gender,
three meteorological variables, binary symptom and comorbidity indicators, and
an 11-class prognosis label. The source has no missing values but contains 219
exact duplicate rows.

One prespecified core symptom, `shivering`, has zero positive observations in
all 5,200 source rows. It is retained in the source/data-quality inventory but
is excluded from model fitting, leaving 36 estimable symptom outcomes.

The stored humidity values range from approximately 0.37 to 1.00. Although the
Zenodo description calls humidity a percentage, this project treats the stored
values as fractions and converts them to percent only inside named
meteorological formulas.

## Important applicability limitation

The released file contains no patient identifier, collection date, location,
weather-station identifier, institution, or independent cohort identifier.
Consequently, patient-level independence, temporal validation, geographic
validation, and the source statement about matched medical records and local
weather stations cannot be independently audited from the file alone.

The primary study is therefore an internal methodological benchmark. It must
not be described as a clinically validated diagnostic or early-warning system.
Clinical claims require an independently collected, temporally ordered,
geographically distinct validation cohort.

## Primary preprocessing policy

1. Preserve the downloaded CSV unchanged.
2. Verify the official MD5 before every publication run.
3. Merge `pain_behind_the_eyes` into `pain_behind_eyes` by row-wise maximum.
4. Remove exact duplicate rows for the primary analysis.
5. Compute engineered meteorological variables from the three raw weather
   variables with explicit units.
6. Fit scaling, thresholds, calibration, and models on development folds only.
7. Keep a locked group-disjoint test set for the final evaluation.
