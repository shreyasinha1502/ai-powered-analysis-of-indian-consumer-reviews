from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
from requests import Session
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "reports"

REQUIRED_COLUMNS = ["review_text", "date", "rating"]
OPTIONAL_COLUMNS = ["verified_purchase", "platform", "brand", "product_name", "price"]
DEFAULT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

CUSTOM_STOPWORDS = {
    "hai",
    "tha",
    "thi",
    "bhi",
    "bahut",
    "bohot",
    "product",
    "item",
    "amazon",
    "flipkart",
    "india",
    "delivery",
    "using",
    "used",
    "buy",
    "purchased",
    "one",
    "get",
}
STOPWORDS = ENGLISH_STOP_WORDS.union(CUSTOM_STOPWORDS)

PRICE_POSITIVE_TERMS = {
    "value",
    "worth",
    "budget",
    "discount",
    "deal",
    "offer",
    "reasonable",
    "affordable",
    "paisa vasool",
}
PRICE_NEGATIVE_TERMS = {
    "expensive",
    "overpriced",
    "costly",
    "high price",
    "pricey",
    "waste of money",
    "not worth",
}
BRAND_POSITIVE_TERMS = {
    "trusted brand",
    "reliable",
    "authentic",
    "original",
    "repeat purchase",
    "loyal",
    "recommend",
    "premium",
}
BRAND_NEGATIVE_TERMS = {
    "fake",
    "duplicate",
    "bad brand",
    "not original",
    "never again",
    "unreliable",
    "damaged",
}

COLUMN_ALIASES = {
    "review": "review_text",
    "review body": "review_text",
    "reviewbody": "review_text",
    "content": "review_text",
    "text": "review_text",
    "review date": "date",
    "review_date": "date",
    "reviewdate": "date",
    "stars": "rating",
    "score": "rating",
    "product": "product_name",
    "product title": "product_name",
    "product_title": "product_name",
    "brand name": "brand",
    "brand_name": "brand",
    "site": "platform",
}

SCRAPER_SELECTORS = {
    "generic": {
        "review": [".review-text", '[data-hook="review-body"]', ".review-content", "p.review-text"],
        "rating": [".review-rating", '[data-hook="review-star-rating"]', ".rating"],
        "date": [".review-date", '[data-hook="review-date"]', ".date"],
        "brand": [".brand", ".review-brand"],
        "product_name": [".product-name", ".review-product"],
    }
}


@dataclass
class SentimentConfig:
    use_transformer: bool = True
    prefer_verified_only: bool = False
    model_name: str = DEFAULT_MODEL_NAME


def ensure_output_dirs() -> None:
    for directory in (FIGURE_DIR, TABLE_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_reviews_dataset(file_path: Path | str) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path)


def scrape_reviews_from_html(
    html_text: str,
    platform: str = "Unknown",
    brand: Optional[str] = None,
    product_name: Optional[str] = None,
    selectors_key: str = "generic",
) -> pd.DataFrame:
    soup = BeautifulSoup(html_text, "html.parser")
    selectors = SCRAPER_SELECTORS[selectors_key]

    def first_text(node, candidates: Sequence[str]) -> str:
        for selector in candidates:
            match = node.select_one(selector)
            if match:
                return match.get_text(" ", strip=True)
        return ""

    rows: List[Dict[str, object]] = []
    for card in soup.select(".review, .review-card, [data-hook='review']"):
        review_text = first_text(card, selectors["review"])
        if not review_text:
            continue

        rating_raw = first_text(card, selectors["rating"])
        rating_match = re.search(r"(\d+(?:\.\d+)?)", rating_raw)
        rows.append(
            {
                "review_text": review_text,
                "date": first_text(card, selectors["date"]) or pd.Timestamp.today().date().isoformat(),
                "rating": float(rating_match.group(1)) if rating_match else None,
                "verified_purchase": "verified" in card.get_text(" ", strip=True).lower(),
                "platform": platform,
                "brand": brand or first_text(card, selectors["brand"]),
                "product_name": product_name or first_text(card, selectors["product_name"]),
            }
        )

    return pd.DataFrame(rows)


def scrape_reviews_from_url(
    url: str,
    platform: str,
    brand: Optional[str] = None,
    product_name: Optional[str] = None,
    session: Optional[Session] = None,
) -> pd.DataFrame:
    session = session or Session()
    response = session.get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        },
        timeout=30,
    )
    response.raise_for_status()
    return scrape_reviews_from_html(
        response.text,
        platform=platform,
        brand=brand,
        product_name=product_name,
    )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for column in df.columns:
        normalized = re.sub(r"\s+", " ", str(column).strip().lower())
        rename_map[column] = COLUMN_ALIASES.get(normalized, normalized.replace(" ", "_"))

    standardized = df.rename(columns=rename_map).copy()
    missing = [column for column in REQUIRED_COLUMNS if column not in standardized.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Expected at least {REQUIRED_COLUMNS}."
        )

    for column in OPTIONAL_COLUMNS:
        if column not in standardized.columns:
            standardized[column] = None

    return standardized


def clean_review_text(text: object) -> str:
    if pd.isna(text):
        return ""

    cleaned = str(text).lower()
    cleaned = re.sub(r"http\S+|www\S+", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9\s']", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def tokenize_text(clean_text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z']+", clean_text)
    return [token for token in tokens if len(token) > 2 and token not in STOPWORDS]


def label_from_compound(score: float) -> str:
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def detect_price_sensitivity(clean_text: str) -> str:
    positive_hits = sum(term in clean_text for term in PRICE_POSITIVE_TERMS)
    negative_hits = sum(term in clean_text for term in PRICE_NEGATIVE_TERMS)
    if positive_hits == 0 and negative_hits == 0:
        return "no_signal"
    if negative_hits > positive_hits:
        return "price_concern"
    if positive_hits > negative_hits:
        return "value_positive"
    return "mixed_signal"


def compute_price_signal_score(clean_text: str) -> int:
    return sum(term in clean_text for term in PRICE_POSITIVE_TERMS) - sum(
        term in clean_text for term in PRICE_NEGATIVE_TERMS
    )


def detect_brand_perception(clean_text: str) -> str:
    positive_hits = sum(term in clean_text for term in BRAND_POSITIVE_TERMS)
    negative_hits = sum(term in clean_text for term in BRAND_NEGATIVE_TERMS)
    if positive_hits == 0 and negative_hits == 0:
        return "neutral_brand_signal"
    if negative_hits > positive_hits:
        return "brand_risk"
    if positive_hits > negative_hits:
        return "brand_positive"
    return "mixed_brand_signal"


def extract_keyword_hits(clean_text: str, keywords: Iterable[str]) -> List[str]:
    return sorted([keyword for keyword in keywords if keyword in clean_text])


def normalize_verified_purchase(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "verified", "y"}


def preprocess_reviews(df: pd.DataFrame, prefer_verified_only: bool = False) -> pd.DataFrame:
    prepared = standardize_columns(df)
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared["rating"] = pd.to_numeric(prepared["rating"], errors="coerce")
    prepared = prepared.dropna(subset=["review_text", "date", "rating"]).copy()

    prepared["review_text"] = prepared["review_text"].astype(str)
    prepared["platform"] = prepared["platform"].fillna("Unknown").astype(str)
    prepared["brand"] = prepared["brand"].fillna("Unknown").astype(str)
    prepared["product_name"] = prepared["product_name"].fillna("Unknown").astype(str)
    prepared["verified_purchase"] = prepared["verified_purchase"].map(normalize_verified_purchase)

    if prefer_verified_only:
        prepared = prepared[prepared["verified_purchase"]].copy()

    prepared["clean_text"] = prepared["review_text"].map(clean_review_text)
    prepared["tokens"] = prepared["clean_text"].map(tokenize_text)
    prepared["token_count"] = prepared["tokens"].map(len)
    prepared["month"] = prepared["date"].dt.to_period("M").astype(str)
    prepared["price_signal"] = prepared["clean_text"].map(detect_price_sensitivity)
    prepared["price_signal_score"] = prepared["clean_text"].map(compute_price_signal_score)
    prepared["brand_perception"] = prepared["clean_text"].map(detect_brand_perception)
    prepared["price_keywords"] = prepared["clean_text"].map(
        lambda text: extract_keyword_hits(text, PRICE_POSITIVE_TERMS.union(PRICE_NEGATIVE_TERMS))
    )
    prepared["brand_keywords"] = prepared["clean_text"].map(
        lambda text: extract_keyword_hits(text, BRAND_POSITIVE_TERMS.union(BRAND_NEGATIVE_TERMS))
    )

    if prepared.empty:
        raise ValueError("No valid reviews remained after preprocessing.")

    return prepared.reset_index(drop=True)


@lru_cache(maxsize=1)
def get_vader_analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


@lru_cache(maxsize=1)
def load_transformer_pipeline(model_name: str):
    from transformers import pipeline

    return pipeline(
        task="sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        truncation=True,
    )


def vader_sentiment(clean_text: str) -> Dict[str, object]:
    analyzer = get_vader_analyzer()
    scores = analyzer.polarity_scores(clean_text or "")
    return {
        "vader_compound": scores["compound"],
        "vader_positive": scores["pos"],
        "vader_neutral": scores["neu"],
        "vader_negative": scores["neg"],
        "vader_label": label_from_compound(scores["compound"]),
    }


def transformer_sentiment(clean_text: str, model_name: str) -> Dict[str, object]:
    if not clean_text:
        return {
            "transformer_label": "neutral",
            "transformer_score": 0.0,
            "transformer_confidence": 0.0,
            "transformer_source": "transformer",
        }

    analyzer = load_transformer_pipeline(model_name)
    result = analyzer(clean_text[:512])[0]
    label_text = str(result["label"]).lower()
    confidence = float(result.get("score", 0.0))

    if "positive" in label_text or label_text.endswith("5 stars") or label_text.endswith("4 stars"):
        label = "positive"
        score = confidence
    elif "negative" in label_text or label_text.endswith("1 star") or label_text.endswith("2 stars"):
        label = "negative"
        score = -confidence
    else:
        label = "neutral"
        score = 0.0

    return {
        "transformer_label": label,
        "transformer_score": score,
        "transformer_confidence": confidence,
        "transformer_source": "transformer",
    }


def combine_sentiment_labels(vader_label: str, transformer_label: str) -> str:
    if vader_label == transformer_label:
        return vader_label
    if "neutral" in {vader_label, transformer_label}:
        return transformer_label if vader_label == "neutral" else vader_label
    return "mixed"


def combine_sentiment_score(vader_compound: float, transformer_score: Optional[float]) -> float:
    if transformer_score is None:
        return round(vader_compound, 4)
    return round((vader_compound + transformer_score) / 2, 4)


def score_sentiment(df: pd.DataFrame, config: SentimentConfig) -> tuple[pd.DataFrame, str]:
    scored = df.copy()
    rows: List[Dict[str, object]] = []
    transformer_available = config.use_transformer

    for row in scored.itertuples(index=False):
        vader_result = vader_sentiment(row.clean_text)
        transformer_result: Dict[str, object]

        if transformer_available:
            try:
                transformer_result = transformer_sentiment(row.clean_text, config.model_name)
            except Exception:
                transformer_available = False
                transformer_result = {
                    "transformer_label": None,
                    "transformer_score": None,
                    "transformer_confidence": None,
                    "transformer_source": "unavailable",
                }
        else:
            transformer_result = {
                "transformer_label": None,
                "transformer_score": None,
                "transformer_confidence": None,
                "transformer_source": "unavailable",
            }

        combined_label = combine_sentiment_labels(
            vader_result["vader_label"],
            transformer_result["transformer_label"] or vader_result["vader_label"],
        )
        combined_score = combine_sentiment_score(
            float(vader_result["vader_compound"]),
            transformer_result["transformer_score"],
        )

        rows.append(
            {
                **vader_result,
                **transformer_result,
                "sentiment_label": combined_label,
                "sentiment_score": combined_score,
            }
        )

    scored = pd.concat([scored.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    model_used = "vader+distilbert" if transformer_available and config.use_transformer else "vader_only"
    return scored, model_used


def build_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    trend = (
        df.groupby("month", as_index=False)
        .agg(
            average_sentiment=("sentiment_score", "mean"),
            vader_average=("vader_compound", "mean"),
            average_rating=("rating", "mean"),
            review_count=("review_text", "size"),
            average_price_signal=("price_signal_score", "mean"),
        )
        .sort_values("month")
    )
    return trend.round(3)


def build_brand_trend(df: pd.DataFrame) -> pd.DataFrame:
    trend = (
        df.groupby(["month", "brand"], as_index=False)
        .agg(
            average_sentiment=("sentiment_score", "mean"),
            review_count=("review_text", "size"),
            price_signal_score=("price_signal_score", "mean"),
        )
        .sort_values(["month", "brand"])
    )
    return trend.round(3)


def build_product_trend(df: pd.DataFrame) -> pd.DataFrame:
    trend = (
        df.groupby(["month", "product_name"], as_index=False)
        .agg(
            average_sentiment=("sentiment_score", "mean"),
            review_count=("review_text", "size"),
        )
        .sort_values(["month", "product_name"])
    )
    return trend.round(3)


def build_price_signal_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["brand", "price_signal"], as_index=False)
        .agg(
            reviews=("review_text", "size"),
            average_sentiment=("sentiment_score", "mean"),
            average_rating=("rating", "mean"),
        )
        .sort_values(["brand", "reviews"], ascending=[True, False])
    )
    return summary.round(3)


def build_brand_perception_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["brand", "brand_perception"], as_index=False)
        .agg(
            reviews=("review_text", "size"),
            average_sentiment=("sentiment_score", "mean"),
        )
        .sort_values(["brand", "reviews"], ascending=[True, False])
    )
    return summary.round(3)


def build_top_terms(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    token_series = df["tokens"].explode().dropna()
    token_counts = token_series.value_counts().head(top_n).reset_index()
    token_counts.columns = ["term", "frequency"]
    return token_counts


def build_topic_model(df: pd.DataFrame, num_topics: int = 4, top_words: int = 8) -> pd.DataFrame:
    documents = df["tokens"].map(lambda tokens: " ".join(tokens))
    documents = documents[documents.str.strip().astype(bool)]
    if documents.empty:
        return pd.DataFrame(columns=["topic_id", "top_terms", "dominant_share"])

    vectorizer = CountVectorizer(max_df=0.95, min_df=1)
    doc_term_matrix = vectorizer.fit_transform(documents)
    topic_count = min(num_topics, max(1, doc_term_matrix.shape[0]))
    lda = LatentDirichletAllocation(n_components=topic_count, random_state=42, learning_method="batch")
    topic_matrix = lda.fit_transform(doc_term_matrix)

    vocabulary = vectorizer.get_feature_names_out()
    rows = []
    for topic_id, weights in enumerate(lda.components_):
        top_indices = weights.argsort()[-top_words:][::-1]
        topic_terms = ", ".join(vocabulary[index] for index in top_indices)
        dominant_share = float(topic_matrix[:, topic_id].mean())
        rows.append(
            {
                "topic_id": f"Topic {topic_id + 1}",
                "top_terms": topic_terms,
                "dominant_share": round(dominant_share, 3),
            }
        )
    return pd.DataFrame(rows)


def _safe_filename(name: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return text or "unknown"


def create_visualizations(
    monthly_trend: pd.DataFrame,
    brand_trend: pd.DataFrame,
    price_summary: pd.DataFrame,
    topic_summary: pd.DataFrame,
) -> None:
    plt.style.use("ggplot")

    plt.figure(figsize=(10, 5))
    plt.plot(monthly_trend["month"], monthly_trend["average_sentiment"], marker="o", linewidth=2, color="#1f77b4")
    plt.title("Monthly Sentiment Trend")
    plt.xlabel("Month")
    plt.ylabel("Average Sentiment Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "monthly_sentiment_trend.png", dpi=300)
    plt.close()

    if not brand_trend.empty:
        plt.figure(figsize=(11, 6))
        for brand, brand_df in brand_trend.groupby("brand"):
            plt.plot(brand_df["month"], brand_df["average_sentiment"], marker="o", linewidth=2, label=brand)
        plt.title("Brand Sentiment Shift Over Time")
        plt.xlabel("Month")
        plt.ylabel("Average Sentiment Score")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "brand_sentiment_trend.png", dpi=300)
        plt.close()

    if not price_summary.empty:
        pivot = (
            price_summary.pivot(index="brand", columns="price_signal", values="reviews")
            .fillna(0)
            .sort_index()
        )
        pivot.plot(kind="bar", stacked=True, figsize=(11, 6), colormap="viridis")
        plt.title("Price Sensitivity Signals by Brand")
        plt.xlabel("Brand")
        plt.ylabel("Review Count")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "price_signal_by_brand.png", dpi=300)
        plt.close()

    if not topic_summary.empty:
        plt.figure(figsize=(10, 5))
        plt.bar(topic_summary["topic_id"], topic_summary["dominant_share"], color="#ff7f0e")
        plt.title("LDA Topic Share")
        plt.xlabel("Topic")
        plt.ylabel("Average Topic Weight")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "topic_share.png", dpi=300)
        plt.close()


def generate_findings_report(
    scored_df: pd.DataFrame,
    monthly_trend: pd.DataFrame,
    brand_trend: pd.DataFrame,
    price_summary: pd.DataFrame,
    brand_summary: pd.DataFrame,
    topic_summary: pd.DataFrame,
    model_used: str,
) -> str:
    weakest_month = monthly_trend.sort_values("average_sentiment").iloc[0]
    strongest_month = monthly_trend.sort_values("average_sentiment", ascending=False).iloc[0]

    brand_shift = pd.DataFrame()
    if not brand_trend.empty:
        brand_shift = (
            brand_trend.sort_values(["brand", "month"])
            .groupby("brand")
            .agg(
                first_sentiment=("average_sentiment", "first"),
                last_sentiment=("average_sentiment", "last"),
            )
            .reset_index()
        )
        brand_shift["sentiment_change"] = brand_shift["last_sentiment"] - brand_shift["first_sentiment"]

    improving_brand = (
        brand_shift.sort_values("sentiment_change", ascending=False).iloc[0]["brand"]
        if not brand_shift.empty
        else "N/A"
    )
    declining_brand = (
        brand_shift.sort_values("sentiment_change").iloc[0]["brand"]
        if not brand_shift.empty
        else "N/A"
    )

    price_focus = price_summary[price_summary["price_signal"].isin(["price_concern", "value_positive"])]
    top_price_brand = price_focus.sort_values("reviews", ascending=False).iloc[0]["brand"] if not price_focus.empty else "N/A"

    informative_brand_summary = brand_summary[brand_summary["brand_perception"] != "neutral_brand_signal"]
    source_brand_summary = informative_brand_summary if not informative_brand_summary.empty else brand_summary
    top_brand_signal = (
        source_brand_summary.sort_values("reviews", ascending=False).iloc[0]["brand_perception"]
        if not source_brand_summary.empty
        else "N/A"
    )
    top_topic = topic_summary.sort_values("dominant_share", ascending=False).iloc[0]["top_terms"] if not topic_summary.empty else "N/A"

    report = f"""# AI-Powered Analysis of Indian Consumer Reviews

## Project Summary
This report analyzes Indian e-commerce reviews from Amazon India and Flipkart style datasets to track:
- sentiment trends over time
- price sensitivity signals
- brand perception shifts

The production pipeline used `{model_used}` for sentiment scoring alongside tokenized text preprocessing and LDA topic modeling.

## Headline Findings
1. Sentiment peaked in `{strongest_month['month']}` with an average score of `{strongest_month['average_sentiment']:.3f}` and weakened most in `{weakest_month['month']}` at `{weakest_month['average_sentiment']:.3f}`.
2. `{improving_brand}` showed the strongest positive brand trajectory over the observed window, while `{declining_brand}` showed the steepest softening in sentiment.
3. `{top_price_brand}` had the highest concentration of explicit price-value discussion, making it the clearest candidate for price-positioning analysis.
4. The most common brand perception bucket was `{top_brand_signal}`, suggesting that brand trust and authenticity are central themes in the review set.
5. Topic modeling surfaced a dominant theme around `{top_topic}`, which can guide merchandising and customer experience prioritization.

## Analytical Notes
- VADER is useful for quick lexical polarity and works well on short reviews.
- DistilBERT improves context sensitivity when transformer weights are available locally.
- Price sensitivity is extracted using review-level value and affordability expressions such as `worth`, `budget`, `overpriced`, and `paisa vasool`.
- Brand perception is inferred from trust, authenticity, loyalty, and fake/duplicate risk cues.

## Deployment
- Local app: `streamlit run indian_ecommerce_nlp\\app.py`
- CLI batch run: `python -m indian_ecommerce_nlp.run_analysis --input indian_ecommerce_nlp\\data\\sample_reviews.csv`
- Deploy on Streamlit Community Cloud, Render, or Hugging Face Spaces with Python 3.11 and `indian_ecommerce_nlp/app.py` as the entrypoint.

## Output Artifacts
- `outputs/tables/scored_reviews.csv`
- `outputs/tables/monthly_sentiment_trend.csv`
- `outputs/tables/brand_sentiment_trend.csv`
- `outputs/tables/product_sentiment_trend.csv`
- `outputs/tables/topic_summary.csv`
- `outputs/figures/*.png`
"""
    return report


def save_outputs(
    scored_df: pd.DataFrame,
    monthly_trend: pd.DataFrame,
    brand_trend: pd.DataFrame,
    product_trend: pd.DataFrame,
    price_summary: pd.DataFrame,
    brand_summary: pd.DataFrame,
    topic_summary: pd.DataFrame,
    top_terms: pd.DataFrame,
    findings_report: str,
) -> None:
    scored_df.to_csv(TABLE_DIR / "scored_reviews.csv", index=False)
    monthly_trend.to_csv(TABLE_DIR / "monthly_sentiment_trend.csv", index=False)
    brand_trend.to_csv(TABLE_DIR / "brand_sentiment_trend.csv", index=False)
    product_trend.to_csv(TABLE_DIR / "product_sentiment_trend.csv", index=False)
    price_summary.to_csv(TABLE_DIR / "price_signal_summary.csv", index=False)
    brand_summary.to_csv(TABLE_DIR / "brand_perception_summary.csv", index=False)
    topic_summary.to_csv(TABLE_DIR / "topic_summary.csv", index=False)
    top_terms.to_csv(TABLE_DIR / "top_terms.csv", index=False)
    (REPORT_DIR / "findings_report.md").write_text(findings_report, encoding="utf-8")


def build_summary_metrics(df: pd.DataFrame) -> Dict[str, object]:
    return {
        "review_count": int(len(df)),
        "avg_rating": round(float(df["rating"].mean()), 2),
        "avg_sentiment": round(float(df["sentiment_score"].mean()), 3),
        "price_signal_share": round(float((df["price_signal"] != "no_signal").mean()) * 100, 1),
        "brand_signal_share": round(float((df["brand_perception"] != "neutral_brand_signal").mean()) * 100, 1),
        "verified_share": round(float(df["verified_purchase"].mean()) * 100, 1),
        "average_tokens": round(float(df["token_count"].mean()), 1),
    }


def run_pipeline(raw_df: pd.DataFrame, config: Optional[SentimentConfig] = None) -> Dict[str, object]:
    config = config or SentimentConfig()
    ensure_output_dirs()

    prepared = preprocess_reviews(raw_df, prefer_verified_only=config.prefer_verified_only)
    scored, model_used = score_sentiment(prepared, config=config)
    monthly_trend = build_monthly_trend(scored)
    brand_trend = build_brand_trend(scored)
    product_trend = build_product_trend(scored)
    price_summary = build_price_signal_summary(scored)
    brand_summary = build_brand_perception_summary(scored)
    top_terms = build_top_terms(scored)
    topic_summary = build_topic_model(scored)

    create_visualizations(monthly_trend, brand_trend, price_summary, topic_summary)
    findings_report = generate_findings_report(
        scored,
        monthly_trend,
        brand_trend,
        price_summary,
        brand_summary,
        topic_summary,
        model_used,
    )
    save_outputs(
        scored,
        monthly_trend,
        brand_trend,
        product_trend,
        price_summary,
        brand_summary,
        topic_summary,
        top_terms,
        findings_report,
    )

    return {
        "prepared_df": prepared,
        "scored_df": scored,
        "monthly_trend": monthly_trend,
        "brand_trend": brand_trend,
        "product_trend": product_trend,
        "price_summary": price_summary,
        "brand_summary": brand_summary,
        "topic_summary": topic_summary,
        "top_terms": top_terms,
        "summary_metrics": build_summary_metrics(scored),
        "findings_report": findings_report,
        "model_used": model_used,
    }
