# Submission and repository checklist

This checklist separates completed technical work from decisions that require
the authors, supervisor, or an external data owner.

## Completed in revision 2.0

- [x] Preserve the official Zenodo CSV unchanged and verify its MD5.
- [x] Record dataset title, creators, version, DOI, and CC BY 4.0 attribution.
- [x] Remove 219 exact duplicates in the primary analysis.
- [x] Reconcile the duplicated pain-behind-the-eyes field.
- [x] Exclude the constant-zero `shivering` target from model estimation.
- [x] Use one unit-explicit implementation for all weather indices.
- [x] Create a weather-signature-disjoint locked test set.
- [x] Train Stage 2 on development out-of-fold Stage-1 predictions.
- [x] Keep threshold selection and model fitting independent of test labels.
- [x] Compare class-prior, direct, cascade, Oracle, context, skip, and
      Stage-2-architecture controls.
- [x] Report classwise metrics, discrimination, calibration, 2,000-replicate
      bootstrap intervals, and exact paired McNemar tests.
- [x] Save the run configuration, environment, split indices, probabilities,
      thresholds, metrics, and figure-generation code.
- [x] Rewrite the Chinese and English papers around the audited results.
- [x] Replace the unrelated bibliography with sources used by this project.
- [x] Add a clinical-use warning and a prespecified external-validation
      protocol.
- [x] Add checksum-verified source-data retrieval and automated GitHub Actions
      checks for the public code release.

## Must be decided before public repository upload

- [ ] Authors and supervisor approve the final author order, corresponding
      author, contribution statement, and conflict-of-interest statement.
- [x] Release the code under the MIT License. The dataset itself is CC BY 4.0;
      that license does not automatically license the project code.
- [x] Download the raw CSV from Zenodo with the reproduction script instead of
      redistributing it in the repository.
- [ ] Add the final repository URL and release DOI to `CITATION.cff` after they
      exist.
- [ ] Remove student email addresses if the target venue or public repository
      should use a corresponding-author address instead.
- [ ] Confirm the target venue's template, page limit, anonymization policy,
      reference style, figure resolution, and supplementary-material policy.
- [ ] Confirm whether an ethics statement or exemption statement is required,
      even though the current analysis uses a public de-identified table.

## Must be completed before any clinical or early-warning claim

- [ ] Obtain an independently collected cohort with patient/encounter IDs,
      timestamps, locations, weather-station metadata, institutions, and
      independently adjudicated outcomes.
- [ ] Freeze the preprocessing, predictors, thresholds, and model
      configurations before accessing external outcomes.
- [ ] Perform a formal external-validation sample-size calculation after
      external class prevalence is known.
- [ ] Evaluate temporal ordering, participant overlap, missingness,
      distribution shift, discrimination, calibration, subgroup stability, and
      decision usefulness.
- [ ] If recalibration is performed, evaluate the updated model on a second
      untouched cohort.

## Suggested upload contents

Include:

- `src/`, `scripts/`, `configs/`, `tests/`
- `results/publication/` and `figures/publication/`
- `DATASET.md`, `EXPERIMENT_PROTOCOL.md`,
  `EXTERNAL_VALIDATION_PROTOCOL.md`, and `VALIDATION_REPORT.md`
- `requirements.txt`, `CITATION.cff`, and the final license
- the final paper source and visually verified PDF

Keep out of the normal Git history:

- `.venv/`, caches, LaTeX auxiliary files, and `tmp/`
- `data/processed/`
- large fitted-model and probability artifacts under
  `artifacts/publication/`; attach them to a versioned release or archive if
  the target platform permits
- legacy ZIP files unless they are required for provenance
