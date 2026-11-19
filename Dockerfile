# Multi-stage build.
#   target "train": runs the data pipeline + training (used by `docker compose --profile train run pipeline`)
#   target "api"  : small, non-root image serving the FastAPI app (the default/last stage; used by
#                   `docker compose up api`, Render and Railway)
FROM python:3.13-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FRAUD_ROOT=/app \
    PYTHONPATH=/app/src:/app
WORKDIR /app

# ---------------------------------------------------------------- training image
FROM base AS train
COPY requirements-api.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY configs/ configs/
COPY src/ src/
# data/, artifacts/ and reports/ are mounted as volumes by docker-compose.yml
CMD ["sh", "-c", "python -m fraud.data && python -m fraud.train"]

# ---------------------------------------------------------------- API image (default)
FROM base AS api
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt
COPY configs/ configs/
COPY src/ src/
COPY app/ app/
COPY artifacts/ artifacts/
RUN useradd --create-home --uid 10001 appuser
USER appuser
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=4)"
# Render, Railway and similar platforms inject $PORT
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
