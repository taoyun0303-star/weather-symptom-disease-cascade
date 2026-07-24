# Initial literature audit and novelty decision

**Review date:** 2026-07-24  
**Purpose:** determine whether the public weather--symptom--disease audit has
a defensible research contribution beyond a single-dataset demonstration.

## Sources reviewed in the first pass

| Source | What it establishes | Consequence for this project |
| --- | --- | --- |
| Koh et al., *Concept Bottleneck Models*, ICML 2020 ([PMLR](https://proceedings.mlr.press/v119/koh20a.html)) | A concept bottleneck can make intermediate concepts available for inspection and intervention. | A symptom bottleneck is not novel by itself; this project must not claim to introduce concept bottleneck models. |
| TRIPOD+AI statement, BMJ 2024 ([article](https://www.bmj.com/content/385/bmj.q902)) | Prediction-model studies need transparent reporting across data, modelling, validation, results, and open-science practice. | The repository should keep a prespecified protocol, complete provenance, split logic, uncertainty, and a claim boundary. |
| PROBAST+AI, BMJ 2025 ([article](https://www.bmj.com/content/388/bmj-2024-082505)) | Applicability and risk of bias must be assessed separately from apparent model performance. | The project must retain its no-clinical-use boundary and document the missing temporal, geographic, institutional, and patient-level structure. |
| Subbaswamy et al., *From Development to Deployment: Dataset Shift, Causality, and Shift-Stable Models in Health AI*, Biostatistics 2020 ([article](https://academic.oup.com/biostatistics/article/21/2/345/5631850)) | Health-AI performance can fail under dataset shift; deployment claims need evidence that extends beyond an internal split. | Group-disjoint internal evaluation is stronger than a random row split, but it is not external validation. |

## What is not novel enough for a standalone paper

The following are useful findings but are not, alone, a research contribution:

1. Applying a standard concept-bottleneck structure to one tabular dataset.
2. Reporting that an imperfect intermediate symptom predictor degrades an
   end-to-end classifier.
3. Comparing common tabular models on a single data source.
4. Rebranding an internal benchmark as a clinical diagnostic study.

## Candidate contribution that may be defensible

The realistic paper direction is a **reproducible evaluation protocol for
intermediate-concept pipelines under structured leakage and signal-quality
constraints**. To become more than a case study, it must show that the
conclusions are not an accident of one released CSV.

The candidate contribution would be:

> A controlled audit showing how intermediate-concept error, repeated or
> grouped inputs, and split design change the apparent value of an
> interpretable bottleneck compared with direct prediction.

This is a methodological hypothesis, not yet an established contribution.

## Evidence required before a paper decision

At least one of the following must be completed:

### Route A 鈥?Multiple lawful datasets

- Identify at least two additional datasets with a defensible input -> concept
  -> label structure.
- Apply the same prespecified evaluation protocol without changing the claim
  to clinical utility.
- Show whether the central error-propagation and split-sensitivity findings
  reproduce or meaningfully differ.

### Route B 鈥?Controlled synthetic or semi-synthetic study

- Create a fully documented generator in which the input-to-concept signal,
  concept-to-label signal, duplicates, and group structure are controlled.
- Use it to test when a bottleneck helps, harms, or only appears to help under
  a leaky split.
- Treat the released weather benchmark as one empirical illustration, not the
  sole evidence.

## Decision at this stage

**Current status: portfolio-grade research artifact, not yet submission-ready
as a standalone paper.**

The next decision point occurs only after a supervisor reviews the literature
matrix and we establish whether Route A or Route B is feasible. If neither is
feasible, the correct output is a transparent technical report and project
portfolio rather than a weak submission.

## Immediate next tasks

1. Expand this first-pass list into a structured literature matrix with exact
   research question, data, split, intermediate representation, endpoint, and
   limitation for each paper.
2. Screen candidate datasets for lawful access, provenance, and a genuine
   input -> concept -> label temporal ordering.
3. Draft the controlled-study protocol before running any new experiment.
