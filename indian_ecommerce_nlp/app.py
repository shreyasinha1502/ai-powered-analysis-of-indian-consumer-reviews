from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from indian_ecommerce_nlp.pipeline import (
    DATA_DIR,
    FIGURE_DIR,
    REPORT_DIR,
    SentimentConfig,
    load_reviews_dataset,
    run_pipeline,
)


def load_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    uploaded_file.seek(0)
    if suffix == ".xlsx":
        return pd.read_excel(BytesIO(uploaded_file.read()))
    return pd.read_csv(BytesIO(uploaded_file.read()))


st.set_page_config(
    page_title="AI-Powered Analysis of Indian Consumer Reviews",
    page_icon="bar_chart",
    layout="wide",
)

st.title("AI-Powered Analysis of Indian Consumer Reviews")
st.write(
    "Analyze Indian e-commerce reviews for sentiment trends, price sensitivity signals, "
    "brand perception shifts, and topic-level themes."
)

with st.sidebar:
    st.header("Pipeline Controls")
    data_source = st.radio("Choose a data source", ["Sample dataset", "Upload CSV/XLSX"], index=0)
    uploaded_file = None
    if data_source == "Upload CSV/XLSX":
        uploaded_file = st.file_uploader("Upload review file", type=["csv", "xlsx"])

    use_transformer = st.toggle("Use DistilBERT sentiment model", value=True)
    verified_only = st.toggle("Verified purchases only", value=False)
    run_button = st.button("Run Review Analysis", type="primary", use_container_width=True)

st.markdown(
    """
### Expected schema
- `review_text`
- `date`
- `rating`

Optional columns:
- `verified_purchase`
- `platform`
- `brand`
- `product_name`
- `price`
"""
)

if run_button:
    if data_source == "Upload CSV/XLSX" and uploaded_file is None:
        st.error("Upload a dataset before running the pipeline.")
    else:
        with st.spinner("Running the NLP pipeline and generating outputs..."):
            if data_source == "Sample dataset":
                raw_df = load_reviews_dataset(DATA_DIR / "sample_reviews.csv")
            else:
                raw_df = load_uploaded_dataset(uploaded_file)

            results = run_pipeline(
                raw_df,
                config=SentimentConfig(
                    use_transformer=use_transformer,
                    prefer_verified_only=verified_only,
                ),
            )

        metrics = results["summary_metrics"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Reviews", f"{metrics['review_count']:,}")
        col2.metric("Average Rating", f"{metrics['avg_rating']:.2f}")
        col3.metric("Average Sentiment", f"{metrics['avg_sentiment']:.3f}")
        col4.metric("Avg Tokens", f"{metrics['average_tokens']:.1f}")

        col5, col6, col7 = st.columns(3)
        col5.metric("Price Signals", f"{metrics['price_signal_share']:.1f}%")
        col6.metric("Brand Signals", f"{metrics['brand_signal_share']:.1f}%")
        col7.metric("Verified Share", f"{metrics['verified_share']:.1f}%")

        st.caption(f"Primary scoring mode used: {results['model_used']}")

        st.subheader("Sentiment Trends Over Time")
        st.dataframe(results["monthly_trend"], use_container_width=True)
        st.image(str(FIGURE_DIR / "monthly_sentiment_trend.png"))

        st.subheader("Brand Perception Shifts")
        st.dataframe(results["brand_trend"], use_container_width=True)
        brand_trend_chart = FIGURE_DIR / "brand_sentiment_trend.png"
        if brand_trend_chart.exists():
            st.image(str(brand_trend_chart))

        st.subheader("Product Trends")
        st.dataframe(results["product_trend"], use_container_width=True)

        st.subheader("Price Sensitivity Signals")
        st.dataframe(results["price_summary"], use_container_width=True)
        price_chart = FIGURE_DIR / "price_signal_by_brand.png"
        if price_chart.exists():
            st.image(str(price_chart))

        st.subheader("Brand Perception Summary")
        st.dataframe(results["brand_summary"], use_container_width=True)

        st.subheader("LDA Topic Modeling")
        st.dataframe(results["topic_summary"], use_container_width=True)
        topic_chart = FIGURE_DIR / "topic_share.png"
        if topic_chart.exists():
            st.image(str(topic_chart))

        st.subheader("Most Frequent Terms")
        st.dataframe(results["top_terms"], use_container_width=True)

        st.subheader("Sample Scored Reviews")
        display_columns = [
            "date",
            "platform",
            "brand",
            "product_name",
            "rating",
            "review_text",
            "price_signal",
            "brand_perception",
            "vader_label",
            "transformer_label",
            "sentiment_label",
            "sentiment_score",
        ]
        st.dataframe(results["scored_df"][display_columns].head(25), use_container_width=True)

        st.subheader("Findings Report")
        st.markdown(results["findings_report"])
        st.caption(f"Saved report: {REPORT_DIR / 'findings_report.md'}")

        st.success("Analysis complete. Tables, figures, and the report were written to `indian_ecommerce_nlp/outputs/`.")
else:
    st.info("Choose the sample dataset or upload your own review file, then run the analysis.")
