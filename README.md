# Auditing an Interpretable Weather--Symptom--Disease Cascade

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

> **Research boundary:** this is a reproducible model-audit benchmark, not a
> clinical diagnostic, an early-warning system, or evidence of a causal effect
> of weather on disease.

This repository audits a two-stage tabular machine-learning pipeline:

```text
weather (+ optional pre-event context) -> predicted symptoms -> disease label
```

The revision removes duplicate records, makes weather units explicit, uses
weather-signature-disjoint evaluation, trains the disease stage on out-of-fold
symptom predictions, and reports uncertainty and calibration. Its central
finding is negative but informative: an interpretable symptom bottleneck does
not improve the available weather-only prediction task because the first-stage
symptom signal is weak.

## Key locked-test findings

| System | Accuracy | Interpretation |
| --- | ---: | --- |
| Class-prior baseline | 19.36% | Reference performance |
| Best direct weather-only model | 20.16% | Comparable to the class-prior baseline |
| Weather -> symptoms -> TabNet | 11.94% | Deployable cascade underperforms the baseline |
| Direct pre-event-context XGBoost | 41.12% | Different question: context is added beyond weather |
| Context -> symptoms -> XGBoost | 39.62% | Bottleneck retains much, but not all, context performance |
| Oracle true symptoms -> TabNet | 96.29% | Upper bound only; not deployable before symptoms are observed |

See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for the full interpretation
and [results/publication/endpoint_metrics.csv](results/publication/endpoint_metrics.csv)
for all reported endpoints and confidence intervals.

## Repository layout

```text
src/weather_health/       Reusable data, model, split, and metric code
scripts/                  Dataset retrieval, experiments, figures, manifests
configs/publication.yaml  Versioned experiment configuration
tests/                    Leakage, data-contract, and report-consistency tests
results/publication/      Versioned tables, metrics, and release manifest
figures/publication/      Publication figures generated from the result tables
```

Legacy coursework files are intentionally excluded from the public release.
They remain in the local working directory for traceability but are not part of
the maintained pipeline.

## Quick start

The project targets Python 3.12.

```powershell
git clone <repository-url>
cd weather-symptom-disease-cascade
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\fetch_dataset.py
.\.venv\Scripts\python.exe -m pytest -q
```

`fetch_dataset.py` downloads the canonical Zenodo CSV to `data/raw/` and
checks its published MD5 before making it available to the pipeline. The CSV is
not committed to this repository by default.

## Reproduce the project workflow

Run these commands from the repository root after completing the quick start:

```powershell
.\.venv\Scripts\python.exe scripts\run_publication_experiments.py --config configs\publication.yaml
.\.venv\Scripts\python.exe scripts\compute_publication_importance.py
.\.venv\Scripts\python.exe scripts\make_publication_figures.py
.\.venv\Scripts\python.exe scripts\build_release_manifest.py
```

The experiment saves model probabilities and split indices to
`artifacts/publication/` (ignored by Git because they are derived, large
artifacts). The checked-in results and figures are sufficient to inspect the
reported findings; rerunning the workflow regenerates all derived artifacts.

## Data, ethics, and scope

The source is the CC BY 4.0 *Weather-related Disease Prediction Dataset* on
Zenodo (DOI: [10.5281/zenodo.11366485](https://doi.org/10.5281/zenodo.11366485)).
Its published MD5 is verified by every full experiment. See [DATASET.md](DATASET.md)
for provenance, processing choices, and the limitations created by missing
patient, time, location, and institution identifiers.

Do not use this repository for clinical decisions. Any clinical or
early-warning claim requires an independently collected, temporally ordered,
externally validated cohort; the required protocol is in
[EXTERNAL_VALIDATION_PROTOCOL.md](EXTERNAL_VALIDATION_PROTOCOL.md).

## Citation

- [Citation metadata](CITATION.cff)

## Research-status records

- [Research fact sheet](RESEARCH_FACT_SHEET.md)
- [Weather-case data audit](DATA_AUDIT.md)
- [Weather-case robustness results](results/robustness/RESULTS.md)
- [Controlled second-validation design](SECOND_VALIDATION_DESIGN.md)
- [Controlled second-validation results](results/synthetic/RESULTS.md)

If you use the code or findings, cite this project and the source dataset. The
source code is released under the [MIT License](LICENSE); the source dataset is
not redistributed and remains available under its own CC BY 4.0 license.

## Contributing and release policy

Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing changes. Every change
to the experimental protocol, reported numbers, or scientific claims must be
accompanied by tests and regenerated evidence.
