from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from weather_health.experiment import PublicationExperiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the leakage-resistant publication experiment protocol."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "publication.yaml"),
        help="Path to the publication YAML configuration.",
    )
    args = parser.parse_args()
    experiment = PublicationExperiment(args.config)
    experiment.run()
    print(f"Publication results written to: {experiment.result_dir}")


if __name__ == "__main__":
    main()
