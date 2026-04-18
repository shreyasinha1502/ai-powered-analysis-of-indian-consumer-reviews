from __future__ import annotations

import argparse
from pathlib import Path

from indian_ecommerce_nlp.pipeline import SentimentConfig, load_reviews_dataset, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AI-powered analysis of Indian consumer reviews."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("indian_ecommerce_nlp/data/sample_reviews.csv"),
        help="Path to the review dataset CSV or XLSX file.",
    )
    parser.add_argument(
        "--disable-transformer",
        action="store_true",
        help="Use VADER only and skip DistilBERT sentiment scoring.",
    )
    parser.add_argument(
        "--verified-only",
        action="store_true",
        help="Keep only verified-purchase reviews.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_df = load_reviews_dataset(args.input)
    results = run_pipeline(
        raw_df,
        config=SentimentConfig(
            use_transformer=not args.disable_transformer,
            prefer_verified_only=args.verified_only,
        ),
    )

    metrics = results["summary_metrics"]
    print("Pipeline complete.")
    print(f"Reviews analyzed: {metrics['review_count']}")
    print(f"Average rating: {metrics['avg_rating']}")
    print(f"Average sentiment: {metrics['avg_sentiment']}")
    print(f"Model mode: {results['model_used']}")


if __name__ == "__main__":
    main()
