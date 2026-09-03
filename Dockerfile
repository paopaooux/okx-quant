FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
ARG HTTP_PROXY
ARG HTTPS_PROXY
ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY}
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml requirements.txt ./
COPY scripts ./scripts
COPY strategies ./strategies

# Install only what the combination backtest needs.  Model training and OKX
# execution are separate local commands; this image does not start either.
RUN pip install --no-cache-dir --disable-pip-version-check \
    "pandas>=2.0,<4" "numpy>=1.26,<3" \
    "scikit-learn>=1.3,<2" "lightgbm>=4,<5" "PyYAML>=6.0" \
    .

ENV HTTP_PROXY= \
    HTTPS_PROXY=

CMD ["python", "-m", "scripts.combinations.run"]
