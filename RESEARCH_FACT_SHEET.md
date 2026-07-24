# Research fact sheet and decision record

**Project:** *Auditing an Interpretable Weather--Symptom--Disease Cascade*  
**Status:** public reproducible model-audit benchmark; paper readiness under evaluation  
**Last reviewed:** 2026-07-24

## 1. Scope and research question

This project asks whether a two-stage, interpretable pipeline

```text
weather (+ optional pre-event context) -> predicted symptoms -> disease label
```

adds value over direct disease prediction under a leakage-resistant evaluation
protocol. The primary focus is error propagation and the
interpretability--performance trade-off introduced by a semantic symptom
bottleneck.

## 2. Public-version authorship and responsibility

The public v2 research release is led, rebuilt, analyzed, and maintained by
**Yun Tao**. On 2026-07-24, the repository maintainer confirmed that the
research question, redesigned audit protocol, analysis, and writing for this
public version are her work. Yun Tao is therefore the sole author recorded in
the public citation metadata for this release.

This attribution applies only to this public, reconstructed research artifact.
It does not erase or make claims about contributions to excluded legacy
coursework. Any future contributor who makes a material contribution to the
research conception, methods, analysis, software, or manuscript must be
credited according to the contribution record in force at that time.

## 3. Evidence inventory

| Item | Current evidence | Status |
| --- | --- | --- |
| Data provenance | Zenodo record `10.5281/zenodo.11366485`, version v1, official MD5 verified before full runs | Verified for the released file |
| Data governance | Source is not redistributed; its CC BY 4.0 terms remain attached to the source record | Documented |
| Leakage controls | Exact duplicates removed; weather-signature-disjoint locked test set; development OOF predictions used for stage 2 | Implemented and tested |
| Reproducibility | Versioned configuration, scripts, result manifest, tests, and passing CI | Implemented |
| Uncertainty | Bootstrap intervals and exact McNemar comparisons for reported endpoints | Implemented |
| External validation | No independent temporal, geographic, institutional, or patient-linked cohort available in the source | Not available |

## 4. Claims supported by the current artifact

The current release supports these bounded claims:

1. Under the prespecified grouped internal evaluation, the available
   weather-only symptom signal is weak.
2. In the released benchmark, the deployable weather-to-predicted-symptoms
   cascade underperforms the strongest direct weather-only baseline.
3. The Oracle experiment shows that observed symptoms contain substantially
   more label information than the weather-only predicted symptom
   representation in this benchmark.
4. Adding documented pre-event context changes the task and improves the
   direct prediction benchmark; the bottleneck retains much, but not all, of
   that context-based performance.

## 5. Claims explicitly not supported

The project must not be described as:

- a clinical diagnostic or treatment system;
- an early-warning or disease-outbreak forecasting system;
- evidence that weather causes disease;
- external validation of a health-AI model;
- evidence of patient-level, temporal, geographic, or institutional
  generalization.

## 6. Paper-readiness gate

The project remains a research-quality portfolio artifact unless all of the
following are satisfied before any manuscript submission:

1. A literature review identifies a defensible methodological gap beyond this
   single dataset audit.
2. The contribution and all wording are reviewed with an appropriate academic
   supervisor.
3. The final claims remain stable under prespecified robustness checks.
4. The manuscript reports the source-data limitations and does not imply
   clinical utility without an independent, lawful validation cohort.
5. Authorship and contribution records are reconfirmed before submission.

## 7. Next evidence priorities

1. Build a literature matrix on semantic bottlenecks, error propagation,
   leakage-resistant health-AI evaluation, and negative-result reporting.
2. Add prespecified robustness and sensitivity analyses only where the source
   variables permit them.
3. Obtain supervisor review before selecting any publication venue.
4. Prepare both a manuscript outline and a portfolio-grade technical report;
   the paper path is chosen only after the readiness gate.

## 8. Change-control rule

Any change to the experimental protocol, reported values, scientific claims,
or authorship must be documented in a pull request with a clear rationale.
No medical, causal, or external-validity claim may be added without supporting
evidence and review.
