FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

WORKDIR /app

COPY indian_ecommerce_nlp/requirements.txt /app/indian_ecommerce_nlp/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/indian_ecommerce_nlp/requirements.txt

COPY . /app

EXPOSE 8501

CMD ["sh", "-c", "streamlit run indian_ecommerce_nlp/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0"]
