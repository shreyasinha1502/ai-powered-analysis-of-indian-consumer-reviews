# AI-Powered Analysis of Indian Consumer Reviews

## Project Summary
This report analyzes Indian e-commerce reviews from Amazon India and Flipkart style datasets to track:
- sentiment trends over time
- price sensitivity signals
- brand perception shifts

The production pipeline used `vader_only` for sentiment scoring alongside tokenized text preprocessing and LDA topic modeling.

## Headline Findings
1. Sentiment peaked in `2026-02` with an average score of `0.565` and weakened most in `2025-11` at `0.004`.
2. `Generic` showed the strongest positive brand trajectory over the observed window, while `boAt` showed the steepest softening in sentiment.
3. `Generic` had the highest concentration of explicit price-value discussion, making it the clearest candidate for price-positioning analysis.
4. The most common brand perception bucket was `brand_positive`, suggesting that brand trust and authenticity are central themes in the review set.
5. Topic modeling surfaced a dominant theme around `brand, feels, premium, cello, trusted, original, quality, looks`, which can guide merchandising and customer experience prioritization.

## Analytical Notes
- VADER is useful for quick lexical polarity and works well on short reviews.
- DistilBERT improves context sensitivity when transformer weights are available locally.
- Price sensitivity is extracted using review-level value and affordability expressions such as `worth`, `budget`, `overpriced`, and `paisa vasool`.
- Brand perception is inferred from trust, authenticity, loyalty, and fake/duplicate risk cues.

## Deployment
- Local app: `streamlit run indian_ecommerce_nlp\app.py`
- CLI batch run: `python -m indian_ecommerce_nlp.run_analysis --input indian_ecommerce_nlp\data\sample_reviews.csv`
- Deploy on Streamlit Community Cloud, Render, or Hugging Face Spaces with Python 3.11 and `indian_ecommerce_nlp/app.py` as the entrypoint.

## Output Artifacts
- `outputs/tables/scored_reviews.csv`
- `outputs/tables/monthly_sentiment_trend.csv`
- `outputs/tables/brand_sentiment_trend.csv`
- `outputs/tables/product_sentiment_trend.csv`
- `outputs/tables/topic_summary.csv`
- `outputs/figures/*.png`
