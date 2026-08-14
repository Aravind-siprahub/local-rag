"""CLI Entry Point: python -m app.eval.run"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.eval.runner import DEFAULT_API_URL, DEFAULT_DATASET_PATH, run_evaluation_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeatable Local RAG Evaluation System")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET_PATH), help="Path to benchmark dataset JSON file")
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL, help="Target chat API URL")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for reports")
    parser.add_argument("--save-baseline", type=str, default=None, help="Path to save evaluation baseline JSON")
    parser.add_argument("--compare", type=str, default=None, help="Path to baseline JSON file for BASELINE vs CURRENT comparison")

    args = parser.parse_args()

    run_evaluation_suite(
        dataset_path=args.dataset,
        api_url=args.api_url,
        output_dir=args.output_dir,
        save_baseline_path=args.save_baseline,
        compare_baseline_path=args.compare,
    )


if __name__ == "__main__":
    main()
