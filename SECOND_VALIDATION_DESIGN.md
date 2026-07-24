# Second validation study: controlled concept-cascade stress test

**Status:** prespecified design; no results have been inspected for this study.  
**Purpose:** test when an interpretable concept cascade can retain, and when it loses, predictive performance relative to a direct model under a known data-generating process.

## Why this study is needed

The weather case study establishes one reproducible negative result: its symptom cascade does not add value over a direct weather-only model. It cannot by itself identify *why* the cascade fails, because the source data lack collection-time, site, patient, and causal-provenance metadata. A controlled study can isolate two mechanisms without making any clinical claim:

1. error introduced while predicting the intermediate concepts; and
2. target-relevant signal in the inputs that is not represented by the concepts.

## Question and predeclared hypotheses

**Question.** Under group-disjoint evaluation, how do concept-prediction noise and an unrepresented direct signal change the gap between a direct classifier and a two-stage concept cascade?

- **H1 — favourable bottleneck:** when the target is fully mediated by accurately recoverable concepts, the cascade should be close to the direct model (predeclared practical tolerance: no more than 2 percentage points lower mean accuracy).
- **H2 — concept noise:** increasing concept-measurement noise should make the cascade fall further behind the direct model.
- **H3 — residual signal:** when inputs contain a target-relevant direct path omitted from the concept set, the direct model should outperform the cascade.

The study is explanatory simulation, not evidence about biology, weather, disease, or clinical prediction.

## Fixed data-generating process

Each generated dataset has 480 groups with 6 observations per group, six binary concepts, three residual input features, six nuisance features, and four outcome classes. Group-level latent variables create within-group dependence. Concepts are sampled from a logistic function of latent factors; observed concept-signal features are noisy measurements of those factors. The outcome is sampled from a multinomial logistic model of the true concepts plus, where applicable, residual features.

| Scenario | Concept-measurement noise | Direct residual path | Intended diagnostic |
|---|---:|---:|---|
| `fully_mediated` | 0.35 | 0.00 | A high-fidelity concept bottleneck |
| `concept_noise` | 1.50 | 0.00 | Pure error-propagation stress test |
| `residual_signal` | 0.35 | 1.00 | Concepts omit target-relevant input information |

No scenario, sample size, model, or threshold may be changed after inspecting the outcomes without creating a new protocol version.

## Evaluation protocol

- Three independently generated datasets, master seeds 2026, 2027, and 2028, per scenario.
- Five `StratifiedGroupKFold` outer folds per generated dataset; no group may appear in both development and test.
- The cascade produces out-of-fold concept probabilities on development rows before fitting its outcome stage. It refits the concept stage on all development rows only to predict the held-out test rows.
- Both the direct outcome model and the cascade outcome model use fixed, regularised multinomial logistic regression. The goal is to compare **information pathways**, not tune architecture capacity.
- Endpoints: class-prior baseline, direct classifier, and concept cascade. Report disease-label accuracy and macro F1; report concept macro F1 for the cascade.
- Results are descriptive fold/seed summaries. We will not treat correlated resampling folds as independent samples for a post-hoc significance claim.

## Decision rules before execution

1. H1 is supported only if the five-fold, three-seed mean direct-minus-cascade accuracy gap in `fully_mediated` is at most 2 percentage points.
2. H2 is supported only if the mean direct-minus-cascade accuracy gap is larger in `concept_noise` than in `fully_mediated`.
3. H3 is supported only if the mean direct-minus-cascade accuracy gap is larger in `residual_signal` than in `fully_mediated`.
4. Any failed run is retained in a failure log; it is not silently rerun or dropped.

## External benchmark gate

Only after the controlled study is complete will we decide whether to reproduce the protocol on CUB-200-2011. CUB is a relevant external concept benchmark because it provides 11,788 bird images, 200 classes, and 312 binary attributes, but its images are restricted to non-commercial research/educational use and its test images may overlap ImageNet pretraining data. A CUB extension must therefore document its data license and avoid or explicitly address pretrained-feature leakage. See the official [CUB dataset page](https://www.vision.caltech.edu/datasets/cub_200_2011/) and the original [Concept Bottleneck Models paper](https://proceedings.mlr.press/v119/koh20a.html).
