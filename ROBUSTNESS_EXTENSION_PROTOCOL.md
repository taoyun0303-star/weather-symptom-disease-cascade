# Robustness extension protocol (v2.1, prespecified)

## Purpose

The existing v2.0 result uses one locked, weather-signature-disjoint outer
test fold. This extension evaluates whether its qualitative conclusion is
stable across additional group-disjoint partitions. It is **not** a new primary
endpoint and will not replace the v2.0 locked-test result.

## Question

Across prespecified weather-signature-disjoint resamples, does the deployable
weather -> predicted-symptom -> disease cascade consistently underperform or
match direct weather-to-disease prediction, and how variable is the observed
gap?

## Data and preprocessing

- Use the same checksum-verified Zenodo v1 CSV and existing v2.0 cleaning
  policy.
- Remove exact duplicate rows and retain the same 36 estimable symptoms.
- Use the existing raw-weather signature as the grouping variable.
- Do not add new covariates, data sources, labels, feature-engineering
  families, or hyperparameter searches.

## Prespecified endpoints

Only these weather-only systems are included:

1. empirical class-prior baseline;
2. direct weather-to-disease XGBoost baseline;
3. XGBoost weather-to-symptom probabilities followed by TabNet disease stage.

The Oracle and documented-context systems are excluded because they answer
different questions from the deployable weather-only comparison.

## Resampling design

### Primary robustness panel

- Construct all five outer folds from `StratifiedGroupKFold(n_splits=5,
  shuffle=True, random_state=2026)`.
- For each outer fold, generate development-only out-of-fold symptom
  predictions using the current five-fold grouped inner procedure.
- Fit all fixed endpoint models on that fold's development data and evaluate
  once on its held-out group-disjoint test partition.

### Seed sensitivity panel

- Repeat the first outer-fold procedure with seeds `2027`, `2028`, and `2029`.
- This panel checks partition sensitivity only; it is not pooled with the
  five-fold panel as if all observations were independent.

## Frozen modelling choices

- Reuse existing v2.0 model families, feature construction, random-state
  derivation, calibration procedure, and hyperparameters.
- No endpoint-specific tuning may use any outer-fold test label.
- If a model fails because a fold lacks required class support, record the
  failure and class-support profile; do not silently retry with a different
  fold or remove the result.

## Outcomes to report

For every endpoint and outer fold, report:

- accuracy, balanced accuracy, macro F1, negative log likelihood, multiclass
  Brier score, and expected calibration error;
- the accuracy and macro-F1 difference between direct XGBoost and the
  deployable cascade;
- class counts, group counts, test-row count, and any unsupported class;
- within-fold bootstrap intervals using the existing resampling convention.

Summarise the panel with fold-level tables and plots showing the distribution
of direct-minus-cascade differences. Do not describe the fold distribution as
external validation or claim formal population-level significance from a small
number of correlated resamples.

## Interpretation rules

1. If direct XGBoost exceeds the cascade in most valid outer folds, report the
   original negative finding as stable within this benchmark.
2. If the direction changes materially across folds, report instability rather
   than selecting the most favourable result.
3. If class support makes a fold infeasible, treat it as a limitation of the
   dataset/grouping structure, not as a reason to change the split after
   results are known.
4. Regardless of outcome, retain the no-clinical-use and no-external-validation
   boundary.

## Outputs and change control

New results will be written under a separate `results/robustness/` path, with
their own manifest, split indices, figures, and report. The existing
`results/publication/` data are immutable reference outputs for v2.0.
