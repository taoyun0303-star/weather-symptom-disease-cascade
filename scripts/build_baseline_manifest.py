from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "artifacts" / "publication" / "baseline_manifest.json"
EXCLUDED_PARTS = {
    ".venv",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "artifacts",
    "results",
    "figures",
    "tmp",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    records = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(PROJECT_ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        records.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Immutable hash inventory of the pre-publication legacy baseline.",
        "python": sys.version,
        "platform": platform.platform(),
        "file_count": len(records),
        "files": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(records)} file hashes to {OUTPUT}")


if __name__ == "__main__":
    main()
