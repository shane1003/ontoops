FROM python:3.12-slim

# opendataloader-pdf runs its bundled jar with `java -jar`; without a JRE,
# /api/v1/parser-odl dies with FileNotFoundError. Placed before the sources to pin the
# layer.
# The major version is pinned directly instead of the metapackage
# (default-jre-headless): when the base Debian moves up, the metapackage silently
# installs a different major JVM, and the build and API still succeed while only the
# parse results change (this already happened once, 17->21 on bookworm->trixie).
# The patch is not pinned: Debian deletes old patches when security updates ship
# (21.0.10 is already gone), so a patch pin is bound to break the build some day.
# Pinning only the major blocks the risk and lets patches flow.
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Take only the uv binary (no separate install needed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Install dependencies first to use the layer cache (avoids reinstalling on source
# changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy sources
COPY . .

EXPOSE 8080

# Deployment run: uvicorn alone, without reload
CMD ["uv", "run", "--no-sync", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
