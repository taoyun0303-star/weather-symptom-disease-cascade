# Contributing

Thank you for improving the reproducibility of this project.

## Scope

This is a methodological audit, not clinical software. Contributions must not
add diagnostic, treatment, causal, or early-warning claims unless they include
independent, appropriately governed external validation.

## Development workflow

1. Use Python 3.12 and install dependencies from `requirements.txt`.
2. Retrieve the canonical dataset with `python scripts/fetch_dataset.py`.
3. Keep all experimental choices in `configs/publication.yaml`; do not hardcode
   reported metrics in figures or manuscripts.
4. Run `python -m pytest -q` before opening a pull request.
5. Regenerate affected result tables, figures, and the release manifest when a
   change affects data processing, model behavior, or reported outcomes.

## Pull requests

Explain the scientific motivation, files changed, validation performed, and any
effect on reported metrics. Do not commit virtual environments, raw data,
trained-model artifacts, personal data, or credentials.

## Issues

Please report reproducibility defects with the command, operating system,
Python version, and traceback. For vulnerabilities, follow
[SECURITY.md](SECURITY.md) instead of publishing sensitive details in an issue.
