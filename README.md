# AI-Powered Analysis of Indian Consumer Reviews

This project builds an NLP pipeline for Indian e-commerce reviews from Amazon India, Flipkart, or similar marketplaces. It focuses on three business questions:

- sentiment trends over time
- price sensitivity signals
- brand perception shifts

The implementation uses Python, Pandas, Hugging Face Transformers, VADER, BeautifulSoup, Matplotlib, and scikit-learn topic modeling.

Note:
The base `requirements.txt` keeps transformer support optional so the project can run in lower-storage environments. If you want live DistilBERT inference locally, also install a supported backend such as `torch`.

## What the pipeline does

1. Loads review data from CSV/XLSX or parses review HTML with BeautifulSoup.
2. Cleans, tokenizes, and stopword-filters review text.
3. Scores sentiment using both VADER and a DistilBERT classifier.
4. Tracks monthly sentiment by overall dataset, brand, and product.
5. Extracts price sensitivity and brand perception cues from review language.
6. Uses LDA topic modeling to surface the main review themes.
7. Exports charts, tables, and a short findings report.

## Input schema

Required columns:

- `review_text`
- `date`
- `rating`

Optional columns:

- `verified_purchase`
- `platform`
- `brand`
- `product_name`
- `price`







## Output artifacts

Running the pipeline writes:

- `outputs/tables/scored_reviews.csv`
- `outputs/tables/monthly_sentiment_trend.csv`
- `outputs/tables/brand_sentiment_trend.csv`
- `outputs/tables/product_sentiment_trend.csv`
- `outputs/tables/price_signal_summary.csv`
- `outputs/tables/brand_perception_summary.csv`
- `outputs/tables/topic_summary.csv`
- `outputs/tables/top_terms.csv`
- `outputs/figures/monthly_sentiment_trend.png`
- `outputs/figures/brand_sentiment_trend.png`
- `outputs/figures/price_signal_by_brand.png`
- `outputs/figures/topic_share.png`
- `outputs/reports/findings_report.md`

## Deployment
### Application Link :
https://ai-powered-analysis-of-indian-consumer-reviews-zncjgljr9osiyxh.streamlit.app/

## Application Preview
![Project Snapshot1](https://github.com/shreyasinha1502/ai-powered-analysis-of-indian-consumer-reviews/blob/main/ai1.PNG)
![Project Snapshot2](https://github.com/shreyasinha1502/ai-powered-analysis-of-indian-consumer-reviews/blob/main/ai2.PNG)
![Project Snapshot3](https://github.com/shreyasinha1502/ai-powered-analysis-of-indian-consumer-reviews/blob/main/ai3.PNG)




## Suggested presentation story

- Use monthly charts to discuss shifts in customer mood.
- Use the price signal summary to explain value-for-money positioning.
- Use brand perception buckets to discuss trust, authenticity, and repeat purchase signals.
- Use topic modeling to show the main consumer pain points and delight drivers.
