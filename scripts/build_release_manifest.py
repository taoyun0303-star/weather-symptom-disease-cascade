from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "results" / "publication" / "release_manifest.json"
INCLUDED_ROOTS = (
    ".github",
    "configs",
    "figures/publication",
    "results/publication",
    "scripts",
    "src",
    "tests",
)
INCLUDED_FILES = (
    ".gitignore",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "DATASET.md",
    "EXPERIMENT_PROTOCOL.md",
    "EXTERNAL_VALIDATION_PROTOCOL.md",
    "README.md",
    "SECURITY.md",
    "VALIDATION_REPORT.md",
    "requirements.txt",
    "pyproject.toml",
    "LICENSE",
)
EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
}
EXCLUDED_SUFFIXES = {
    ".pyc",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_files() -> list[Path]:
    candidates: set[Path] = set()
    for root in INCLUDED_ROOTS:
        path = PROJECT_ROOT / root
        if path.is_file():
            candidates.add(path)
        elif path.exists():
            candidates.update(
                candidate for candidate in path.rglob("*") if candidate.is_file()
            )
    candidates.update(
        PROJECT_ROOT / relative
        for relative in INCLUDED_FILES
        if (PROJECT_ROOT / relative).is_file()
    )
    return sorted(
        path
        for path in candidates
        if path.resolve() != OUTPUT.resolve()
        if not any(part in EXCLUDED_PARTS for part in path.parts)
        and path.suffix.lower() not in EXCLUDED_SUFFIXES
    )


def main() -> None:
    records = []
    for path in release_files():
        relative = path.relative_to(PROJECT_ROOT)
        records.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Integrity inventory for the public project release.",
        "dataset": {
            "doi": "10.5281/zenodo.11366485",
            "expected_md5": "4aa51b2bb76b45b2000ce71517a0fd1e",
            "distribution": "Fetched from Zenodo; not included in this repository release.",
        },
        "file_count": len(records),
        "files": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} release hashes to {OUTPUT}")


if __name__ == "__main__":
    main()
