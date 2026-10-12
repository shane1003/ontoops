# Unified image: FastAPI server (app.main: conceptualizer + /tbox=tboxcraft) + two CLIs.
# Reproducibility: base and uv are pinned to fixed versions instead of floating (latest)
# tags. For a stronger guarantee, replace the tags with sha256 digests
# (e.g. python:3.12.8-slim-bookworm@sha256:...).
FROM python:3.12.8-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    # JVM path for HermiT (owlready2) reasoning; used by the tboxcraft module.
    TBOXCRAFT_JAVA_EXE=/usr/bin/java

# HermiT reasoning (owlready2) needs a JRE (tboxcraft). Without it only the reasoner
# degrades, but the production image includes it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Copy only the uv (dependency manager) binaries; a fixed version instead of floating
# latest.
COPY --from=ghcr.io/astral-sh/uv:0.7.0 /uv /uvx /bin/

WORKDIR /app

# 1) Dependency layer: separate from the sources to use the cache.
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --no-install-project

# 2) Application sources (app package) + config
COPY app ./app
COPY config ./config
COPY README.md ./
RUN uv sync --no-dev

EXPOSE 8602

# Default: FastAPI server (uvicorn). Classification needs an LLM key (OPENAI_API_KEY; a
# dummy value works for a local server). The /tbox routes need TBOXCRAFT_DATABASE_URL for
# their DB features (without it only those routes return 503).
# The CLIs can still be used (override CMD):
#   docker run --rm conceptualizer:latest conceptualizer extract data/cq/....json
#   docker run --rm conceptualizer:latest tboxcraft update --components components.json
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8602"]
