# Deployment Guide

## Local run

```powershell
cd "C:\Users\hii\Documents\New project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r indian_ecommerce_nlp\requirements.txt
streamlit run indian_ecommerce_nlp\app.py
```

Use [sample_reviews.csv](/C:/Users/hii/Documents/New%20project/indian_ecommerce_nlp/data/sample_reviews.csv) for the first demo run.

## Optional transformer setup

If you want DistilBERT inference instead of the automatic VADER fallback, install a backend such as:

```powershell
pip install torch
```

The code already falls back to `vader_only` mode if transformer dependencies are unavailable.

## Streamlit Community Cloud

- App file: `indian_ecommerce_nlp/app.py`
- Python version: `3.11`
- Install command: `pip install -r indian_ecommerce_nlp/requirements.txt`
- Optional extra packages: `torch`
- Start command: `streamlit run indian_ecommerce_nlp/app.py --server.port $PORT --server.address 0.0.0.0`

## Render or Hugging Face Spaces

Use the same dependency setup and launch command. The project writes output CSVs, figures, and a markdown report into `indian_ecommerce_nlp/outputs/`.

## Verification checklist

After deployment or a local run, confirm these files exist:

- [scored_reviews.csv](/C:/Users/hii/Documents/New%20project/indian_ecommerce_nlp/outputs/tables/scored_reviews.csv)
- [monthly_sentiment_trend.csv](/C:/Users/hii/Documents/New%20project/indian_ecommerce_nlp/outputs/tables/monthly_sentiment_trend.csv)
- [brand_sentiment_trend.csv](/C:/Users/hii/Documents/New%20project/indian_ecommerce_nlp/outputs/tables/brand_sentiment_trend.csv)
- [topic_summary.csv](/C:/Users/hii/Documents/New%20project/indian_ecommerce_nlp/outputs/tables/topic_summary.csv)
- [findings_report.md](/C:/Users/hii/Documents/New%20project/indian_ecommerce_nlp/outputs/reports/findings_report.md)
