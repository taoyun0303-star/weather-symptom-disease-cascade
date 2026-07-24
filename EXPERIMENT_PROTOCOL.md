# Publication experiment protocol, version 2.0

## Prespecified objective

Evaluate whether an interpretable two-stage weather-to-symptom-to-disease
cascade provides useful disease classification performance relative to direct
weather-to-disease baselines, while measuring error propagation from the
symptom layer.

## Split and leakage controls

- Exact duplicate rows are removed for the primary analysis.
- A weather signature constructed from the three raw meteorological variables
  is used as the group identifier.
- One fold from a seeded five-fold stratified group split is locked as the
  final test set.
- All threshold selection and model comparison use only the remaining
  development set.
- Development out-of-fold predictions are generated with five-fold stratified
  group cross-validation.
- The disease-stage model is trained on out-of-fold symptom predictions, not on
  in-sample predictions.
- The locked test labels are never passed to model fitting, threshold
  selection, feature fitting, early stopping, or model selection code.
- A naive stratified row split with the same test-set size is retained only as
  a split-sensitivity diagnostic. Its weather- and symptom-signature overlap
  is compared with the primary grouped split; it does not produce a competing
  endpoint estimate or change the locked test set.

## Prespecified model families

Stage 1, weather to 36 estimable symptoms:

- logistic regression
- multilabel MLP
- XGBoost

The source field `shivering` is not modeled because it is constant zero in the
entire released dataset. Treating it as a learned outcome would be
mathematically undefined and would distort macro-averaged metrics.

Direct weather-to-disease baselines:

- empirical class-prior dummy classifier
- multinomial logistic regression
- random forest
- MLP
- XGBoost

Stage 2:

- TabNet is the prespecified primary disease-stage model
- logistic regression, random forest, MLP, and XGBoost are secondary
  architecture controls for the XGBoost symptom-probability representation

## Prespecified cascade comparisons

1. True symptoms to TabNet: oracle upper bound.
2. XGBoost symptom probabilities to TabNet: primary deployable cascade.
3. XGBoost thresholded symptoms to TabNet: information-loss ablation.
4. Logistic-regression and MLP symptom probabilities to TabNet: stage-1
   sensitivity.
5. XGBoost symptom probabilities to non-TabNet stage-2 models: stage-2
   architecture sensitivity.
6. Direct weather-to-disease models: necessity of the cascade.
7. Context extension: age and gender, followed by age, gender, and documented
   pre-existing conditions.
8. Skip connection: pre-event context concatenated with predicted symptom
   probabilities before the disease classifier.

## Metrics

Stage 1:

- micro and macro F1
- micro and macro average precision
- Hamming loss
- per-symptom prevalence, precision, recall, F1, average precision, and
  development-selected threshold

Disease and end-to-end prediction:

- accuracy
- balanced accuracy
- macro and weighted F1
- macro one-vs-rest AUROC and average precision
- negative log likelihood
- multiclass Brier score
- expected calibration error
- per-class precision, recall, and F1
- confusion matrix

The locked-test accuracy and macro F1 receive 95% bootstrap confidence
intervals. Pairwise endpoint comparisons use McNemar's exact test, reported
with the discordant counts and without converting a non-significant result into
an equivalence claim.

## Interpretation gate

Higher performance is not an acceptance criterion. The acceptance criterion is
a reproducible, leakage-free estimate with uncertainty. If direct baselines
outperform the cascade, the paper must present the cascade as an
interpretability-performance trade-off rather than claiming superiority.
