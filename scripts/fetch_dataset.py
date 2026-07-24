"""Download and checksum-verify the canonical Zenodo dataset."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = (
    "https://zenodo.org/records/11366485/files/"
    "Weather-related%20disease%20prediction.csv?download=1"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "Weather-related disease prediction.csv"
DEFAULT_MD5 = "4aa51b2bb76b45b2000ce71517a0fd1e"
CHUNK_SIZE = 1024 * 1024


def md5sum(path: Path) -> str:
    """Return the lowercase MD5 checksum for one file."""

    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(url: str, output: Path, expected_md5: str, force: bool) -> None:
    """Download to a temporary file, verify it, and atomically replace output."""

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        observed_md5 = md5sum(output)
        if observed_md5 == expected_md5:
            print(f"Dataset already verified: {output}")
            return
        if not force:
            raise RuntimeError(
                f"Existing file has MD5 {observed_md5}, expected {expected_md5}. "
                "Use --force only after checking the file provenance."
            )

    request = Request(url, headers={"User-Agent": "weather-health-research/2.0"})
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, dir=output.parent, suffix=".download"
        ) as temporary:
            temporary_path = Path(temporary.name)
            with urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, temporary, length=CHUNK_SIZE)
            temporary.flush()
            os.fsync(temporary.fileno())
            observed_md5 = md5sum(temporary_path)
            if observed_md5 != expected_md5:
                raise RuntimeError(
                    f"Downloaded file has MD5 {observed_md5}, expected {expected_md5}."
                )
        # NamedTemporaryFile must be closed before os.replace on Windows.
        temporary_path.replace(output)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    print(f"Downloaded and verified dataset: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the canonical Weather-related Disease Prediction Dataset."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Dataset download URL.")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Destination CSV path."
    )
    parser.add_argument(
        "--expected-md5", default=DEFAULT_MD5, help="Expected source-file MD5 checksum."
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing checksum-mismatched file."
    )
    args = parser.parse_args()
    download_dataset(args.url, args.output, args.expected_md5.lower(), args.force)


if __name__ == "__main__":
    main()
