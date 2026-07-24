# Robustness extension results (v2.1)

**Run completed:** 2026-07-24 08:29 UTC  
**Code version:** `caaaad2b0780fa8dd0a6679eeeaf3ca1cd74de62`  
**Protocol:** [`ROBUSTNESS_EXTENSION_PROTOCOL.md`](../../ROBUSTNESS_EXTENSION_PROTOCOL.md)  
**Dataset:** 4,981 analysis rows after removal of 219 exact duplicate rows; verified MD5 `4aa51b2bb76b45b2000ce71517a0fd1e`.

## Status

The prespecified robustness run completed all 24 endpoint evaluations (8 split specifications x 3 endpoints) with **zero recorded failures**. The machine-readable per-run results are in `endpoint_metrics.csv`; the exact split membership is in `split_manifest.csv`.

## Prespecified comparison

The experiment compares three disease-label endpoints without adding features or tuning models after inspecting the outcomes:

1. **Class prior:** a trivial majority-probability baseline.
2. **Direct XGBoost:** weather features directly predict the disease label.
3. **Cascade XGBoost -> TabNet:** weather features predict symptoms; out-of-fold predicted symptoms then predict the disease label.

The primary panel uses all five outer folds from `StratifiedGroupKFold` with split seed 2026. The sensitivity panel reruns outer fold 0 with split/model seeds 2027, 2028, and 2029. The protocol locks this design before execution.

## Aggregate results

Values below are arithmetic means across independently held-out test folds/seeds. They are descriptive summaries, not pooled confidence intervals.

| Panel | Endpoint | Runs | Accuracy | Balanced accuracy | Macro F1 | NLL | ECE (10-bin) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Five-fold primary | Class prior | 5 | 19.43% | 9.09% | 2.96% | 2.273 | 0.001 |
| Five-fold primary | Direct XGBoost | 5 | 20.10% | 12.99% | 12.38% | 2.258 | 0.092 |
| Five-fold primary | Cascade XGBoost -> TabNet | 5 | 12.25% | 9.33% | 4.23% | 2.389 | 0.027 |
| Seed sensitivity | Class prior | 3 | 19.43% | 9.09% | 2.96% | 2.274 | 0.001 |
| Seed sensitivity | Direct XGBoost | 3 | 20.90% | 13.64% | 13.05% | 2.265 | 0.088 |
| Seed sensitivity | Cascade XGBoost -> TabNet | 3 | 11.74% | 9.69% | 3.57% | 2.391 | 0.024 |

Across the primary folds, direct-XGBoost accuracy ranged from **18.05% to 22.09%**, while cascade accuracy ranged from **11.35% to 13.76%**. Across the three seed-sensitivity runs, direct-XGBoost accuracy ranged from **17.95% to 22.59%** and cascade accuracy from **9.53% to 15.65%**.

## Interpretation

1. **The cascade did not add predictive value.** It was lower than Direct XGBoost on mean accuracy, balanced accuracy, macro F1, and NLL in both prespecified panels. This conclusion is stable across all five outer folds and the three seed perturbations.
2. **The direct weather-only signal is weak.** Its mean primary accuracy is only 0.67 percentage points above the class-prior baseline. Its macro F1 and balanced accuracy improve over the trivial baseline, but its calibration error is materially larger. This is not evidence for a useful clinical prediction model.
3. **This is a method-audit result, not a medical claim.** The source dataset has no patient identifiers, time, location, site, or external cohort. These experiments therefore cannot support diagnostic, early-warning, causal, or generalisation claims.

## Consequence for the research direction

This single-dataset case study is now a reproducible and informative **negative result**: an apparently interpretable weather -> symptom -> disease cascade does not outperform a simpler direct baseline under a grouped, out-of-fold evaluation protocol. It is strong portfolio evidence of research practice, but it is **not yet a standalone paper**.

To make a paper-worthy contribution, the next phase must test a general methodological claim on multiple lawful datasets or controlled semi-synthetic data, with the same protocol and explicitly stated data-generating assumptions. See [`LITERATURE_AUDIT.md`](../../LITERATURE_AUDIT.md) and [`DATA_AUDIT.md`](../../DATA_AUDIT.md).
