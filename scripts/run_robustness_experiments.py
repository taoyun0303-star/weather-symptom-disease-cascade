"""Run the prespecified group-disjoint robustness extension."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from weather_health.robustness import RobustnessExperiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the prespecified weather-health robustness extension."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "robustness.yaml"),
        help="Path to the robustness YAML configuration.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and save split manifests without fitting models.",
    )
    args = parser.parse_args()
    results = RobustnessExperiment(args.config).run(dry_run=args.dry_run)
    if args.dry_run:
        print("Robustness split manifest written without fitting models.")
    else:
        print(f"Robustness endpoint rows written: {len(results)}")


if __name__ == "__main__":
    main()
