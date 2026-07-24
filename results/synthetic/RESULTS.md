# Controlled concept-cascade stress test: results

**Run completed:** 2026-07-24  
**Protocol:** [`SECOND_VALIDATION_DESIGN.md`](../../SECOND_VALIDATION_DESIGN.md)  
**Status:** all 45 prespecified specifications completed; 135 endpoint rows; zero failed specifications.

## Aggregate results

Each cell is the descriptive mean across 3 independently generated datasets and 5 group-disjoint outer folds (15 held-out evaluations per endpoint). These are not treated as independent samples for a post-hoc significance claim.

| Scenario | Endpoint | Accuracy | Macro F1 | Concept macro F1 |
|---|---|---:|---:|---:|
| Fully mediated | Class-prior baseline | 29.09% | 11.26% | — |
| Fully mediated | Direct logistic | 36.02% | 32.07% | — |
| Fully mediated | Concept cascade | 36.61% | 31.89% | 66.19% |
| Concept noise | Class-prior baseline | 29.09% | 11.26% | — |
| Concept noise | Direct logistic | 31.89% | 27.09% | — |
| Concept noise | Concept cascade | 32.23% | 26.10% | 58.46% |
| Residual signal | Class-prior baseline | 29.48% | 11.38% | — |
| Residual signal | Direct logistic | 54.25% | 51.29% | — |
| Residual signal | Concept cascade | 34.59% | 29.70% | 66.35% |

## Prespecified decision rules

### H1 — favourable bottleneck

**Supported.** In the fully mediated setting, Direct minus Cascade mean accuracy was **-0.59 percentage points** (36.02% minus 36.61%). The cascade is within the prespecified 2-point practical tolerance, and marginally higher on accuracy.

### H2 — concept noise

**Not practically supported as a cascade-loss effect.** Stage-1 concept macro F1 fell from 66.19% to 58.46% as expected. The literal Direct-minus-Cascade gap increased by only **0.24 percentage points** (-0.59 to -0.35), so it meets the protocol's bare ordering rule, but the cascade remained slightly ahead on accuracy rather than falling behind the direct model. We therefore do not claim that this noise level materially harms cascade task performance relative to direct prediction.

### H3 — residual signal

**Supported strongly.** When target-relevant input information was deliberately omitted from the concept set, Direct minus Cascade mean accuracy was **+19.65 percentage points** (54.25% versus 34.59%). The direct model retained the residual pathway; the cascade could not use it.

## Combined interpretation with the weather case study

The two studies support a narrower, defensible claim:

> A concept cascade is not automatically an accuracy-improving route to interpretability. It can retain performance when concepts cover the relevant information, but it can lose sharply when the concept vocabulary omits target-relevant input signal. Stage-1 concept accuracy alone is not sufficient to infer downstream task value.

The weather case establishes a real-world audit in which the cascade underperformed. The controlled study shows that this should not be generalized into a universal anti-cascade statement: the information coverage of the concept layer is a critical boundary condition.

## Next research gate

The next substantive extension is an external concept-labelled benchmark, not further tuning of these synthetic settings. CUB-200-2011 is the leading candidate, subject to its non-commercial research/education data restriction and an explicit pretrained-feature leakage policy. No CUB data, pretrained feature archive, or model checkpoint is included in this repository.
