FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY scripts ./scripts

# The package metadata keeps the live image small; research-only dependencies
# in requirements.txt are intentionally not installed here.
RUN pip install --no-cache-dir --disable-pip-version-check .

CMD ["python", "-m", "scripts.live.sync_loop"]
