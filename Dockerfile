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
COPY pyproject.toml requirements.txt requirements-runtime.lock ./
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements-runtime.lock
COPY scripts ./scripts
COPY strategies ./strategies

# Keep model/runtime versions fixed when deploying a trading-rule change.
RUN pip install --no-cache-dir --disable-pip-version-check --no-deps .

ENV HTTP_PROXY= \
    HTTPS_PROXY=

CMD ["python", "-m", "scripts.combinations.run"]
