#!/usr/bin/env python3
"""Export trained model from MLflow for deployment."""

from pathlib import Path
import sys

import mlflow.sklearn

import mlflow

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from churn.logger import DEFAULT_LOGGER


def export_model(run_id: str, output_dir: str = "models/random_forest") -> None:
    """Export model from MLflow run to local directory."""
    logger = DEFAULT_LOGGER

    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("http://localhost:5000")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting model from run {run_id} to {output_path}")

        # Download model artifacts
        mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="random_forest", dst_path=str(output_path)
        )

        logger.info(f"Model exported successfully to {output_path}")

        # List exported files
        for file_path in output_path.rglob("*"):
            if file_path.is_file():
                logger.info(f"  {file_path.relative_to(output_path)}")

    except Exception as e:
        logger.exception(f"Failed to export model: {e}")
        raise


def main() -> None:
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Export model from MLflow")
    parser.add_argument("run_id", help="MLflow run ID")
    parser.add_argument(
        "--output-dir",
        default="models/random_forest",
        help="Output directory for model (default: models/random_forest)",
    )

    args = parser.parse_args()

    export_model(args.run_id, args.output_dir)


if __name__ == "__main__":
    main()
