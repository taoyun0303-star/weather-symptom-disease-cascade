# Validation report

## Overall assessment: needs external validation before clinical claims

The revised internal analysis is methodologically suitable for a
proof-of-concept or model-audit project. It is not sufficient for a clinical
prediction claim because the source file does not expose patient identifiers,
timestamps, locations, institutions, or an external cohort.

## Methodology review

- Raw-data MD5 matches the official Zenodo record.
- The primary dataset removes 219 exact duplicates.
- Humidity is explicitly treated as a fraction in the source and converted to
  percent only inside named formulas.
- Training and inference share one feature-construction function.
- Weather signatures are group-disjoint between development and held-out test
  sets.
- A same-size naive stratified row split would share 280 raw-weather
  signatures across development and test, affecting 377 of 997 test rows; the
  primary grouped split reduces both counts to zero.
- Stage-1 thresholds use development out-of-fold predictions only.
- Stage 2 is trained on out-of-fold predicted symptoms for deployable cascade
  experiments.
- The held-out labels are used only by metric and statistical-evaluation code.
- Accuracy and macro F1 include 2,000-replicate bootstrap intervals.
- Pairwise primary comparisons include exact McNemar tests.

## High-impact findings

1. `shivering` is constant zero and cannot be modeled; 36 symptoms are
   estimable.
2. Weather-only symptom prediction is weak: locked-test XGBoost micro F1 is
   0.1889.
3. The class-prior disease baseline reaches 19.36% accuracy, while the best
   direct weather-only model reaches 20.16%.
4. The weather-only XGBoost-to-TabNet cascade reaches 11.94%, below the
   class-prior baseline.
5. The true-symptom Oracle reaches 96.29%, showing that error propagation at
   the semantic bottleneck, rather than lack of disease information in true
   symptoms, is the dominant limitation.
6. Adding age, gender, and pre-existing conditions raises direct XGBoost
   accuracy to 41.12%; a context-aware XGBoost symptom bottleneck retains
   39.62%.

## Required caveats

- Internal group-disjoint validation is not external validation.
- The source description's medical-record/weather-station matching cannot be
  independently verified from the released columns.
- Accuracy must not be interpreted as diagnostic utility.
- Feature attribution is associational and not causal.
- The context extension changes the scientific question from weather-only
  prediction to prediction using pre-event patient context.
