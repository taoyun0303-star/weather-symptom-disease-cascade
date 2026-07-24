# External validation protocol

## Purpose

The current Zenodo file cannot provide temporal, geographic, institutional, or
patient-level external validation. This protocol defines the minimum evidence
needed before the model can be described as clinically or operationally
validated.

## Required cohort fields

- anonymized patient or encounter identifier
- observation timestamp and prediction timestamp
- location or weather-station identifier
- institution/source identifier
- age and sex/gender with documented coding
- pre-existing conditions known before the prediction time
- meteorological measurements with units and station/source metadata
- symptoms observed after or at the prediction time, with definitions
- independently adjudicated disease outcome and outcome time

## Temporal ordering

All model predictors must be available at or before the declared prediction
time. Disease labels, treatments, post-diagnosis symptoms, and measurements
recorded after the outcome must never enter the predictor set.

## Recommended design

1. Freeze the current preprocessing, feature list, thresholds, and model
   configuration before accessing external outcomes.
2. Select a cohort from a later period, different region, or different
   institution.
3. Map external variables to the frozen data dictionary without refitting on
   external labels.
4. Report inclusion/exclusion flow, missingness, class prevalence, and
   predictor distribution shift.
5. Evaluate discrimination and calibration with 95% confidence intervals.
6. Report performance by clinically meaningful age and sex/gender strata when
   sample sizes permit.
7. Recalibration, if needed, must be reported as a separate updated-model
   experiment, followed by a second untouched evaluation cohort.

## Minimum sample-size target

The target should be justified from expected class prevalence and desired
confidence-interval width. As a practical floor, every disease class should
have enough positive outcomes to support stable class-specific sensitivity,
precision, and calibration estimates. A formal validation sample-size
calculation must be completed once external prevalence is known.

## Acceptance gate

No fixed accuracy threshold is prespecified. Acceptance requires:

- complete provenance and temporal ordering
- zero participant overlap with development data
- no outcome-dependent preprocessing
- confidence intervals narrow enough for the intended use
- calibration compatible with the intended decision threshold
- limitations and subgroup instability reported without selective omission
