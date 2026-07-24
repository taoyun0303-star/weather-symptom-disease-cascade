"""Project path helpers used by scripts and the publication pipeline."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]


def resolve_project_path(value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a configured path against the repository root.

    The publication YAML intentionally stores portable relative paths.  Resolving
    them here, rather than against the caller's current directory, makes the
    scripts behave identically from PowerShell, CI, and an IDE.
    """

    path = Path(value)
    return path if path.is_absolute() else project_root / path
