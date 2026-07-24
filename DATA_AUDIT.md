# Data and evaluation audit

**Audit date:** 2026-07-24  
**Scope:** public v2 code, documentation, source-record metadata, and tests.

## Verified facts

| Area | Verified observation | Evidence |
| --- | --- | --- |
| Source record | Zenodo v1 publishes one CSV, DOI `10.5281/zenodo.11366485`, with the recorded MD5 `4aa51b2bb76b45b2000ce71517a0fd1e`. | Zenodo metadata; `scripts/fetch_dataset.py`; `DATASET.md` |
| File integrity | Full experiment loading rejects a checksum mismatch. | `load_publication_data`; `test_raw_checksum_and_primary_profile` |
| Duplicate handling | The public pipeline removes 219 exact duplicate rows, leaving 4,981 rows for the primary analysis. | `load_publication_data`; data-pipeline test |
| Humidity handling | Stored humidity is validated as a fraction in `[0, 1]`; the code rejects values formatted as percentages at the API boundary. | `compute_weather_indices`; humidity tests |
| Primary split | Exact raw-weather triples are hashed into groups; no exact weather signature is shared between development and locked test data or inside development folds. | `make_publication_split`; split-protocol tests |
| Cascade fitting | Stage 2 receives development out-of-fold symptom predictions rather than in-sample symptom predictions. | `PublicationExperiment`; experiment protocol |
| Documentation consistency | Tests guard the headline endpoint values shown in the README and validation report. | `test_report_consistency.py` |

## What this audit does **not** verify

The public source file lacks patient identifiers, dates, locations, weather
station identifiers, institutions, and a separate external cohort. Therefore
this audit cannot verify:

- whether rows are independent patients or repeated observations;
- whether weather was observed before the symptoms or prognosis label;
- whether a record was actually linked to a particular weather station;
- temporal, geographic, or institutional generalization;
- clinical label quality, annotation process, or clinical utility.

The source record's description of anonymised medical records and local weather
stations is reported as source metadata, not independently established fact.

## Evaluation strengths

1. The project does not use an ordinary random row split as its primary
   endpoint.
2. The primary split prevents *exact* raw-weather signatures from crossing the
   development/test boundary.
3. Threshold fitting and stage-2 training use development-only out-of-fold
   predictions.
4. Locked-test accuracy and macro F1 are accompanied by bootstrap intervals,
   and selected comparisons use exact McNemar tests.

## Remaining risks and the required response

| Risk | Why it matters | Required response |
| --- | --- | --- |
| One fixed outer group split | A single locked fold gives one estimate, not the stability of the conclusion across plausible group partitions. | Add a separately labelled repeated grouped-resampling robustness study; never overwrite the locked primary endpoint. |
| Exact-match grouping only | Exact weather triples prevent literal repeats but do not establish separation of near-identical weather states or unknown patient clusters. | Report this precisely as exact-signature separation; do not call it patient-, temporal-, or geographic-disjoint validation. |
| Unknown source-generation process | Very high Oracle performance and repeated feature patterns may reflect the released benchmark's construction rather than a clinically realistic task. | Treat the dataset as a methodological benchmark; do not make clinical performance claims. |
| Context timing not independently auditable | Age, gender, and listed conditions may be plausible pre-event variables, but the public file cannot establish their collection time. | Label these as *documented context variables* unless timing can be verified from source materials. |
| No external cohort | Internal performance cannot establish transportability. | Retain the no-deployment boundary; seek an independent lawful dataset only for a future robustness study. |

## Immediate protocol amendment to consider

The next analysis should be a **separate robustness extension**, not a
replacement for the locked primary experiment:

1. Generate repeated stratified group splits across prespecified seeds.
2. Re-run a reduced, fixed set of direct and cascade endpoints on each split.
3. Report the distribution of performance differences and the number of usable
   classes/groups per split.
4. Keep the existing 2026 locked-test result visible as the original primary
   endpoint and label the new output as robustness evidence.

This extension improves methodological confidence but still does not create
external clinical validation.
