FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Pinned, not :latest. Every Python dependency below is pinned by uv.lock
# --frozen; leaving the tool that reads that lock floating would make the build
# reproducible in its inputs and not in its resolver.
COPY --from=ghcr.io/astral-sh/uv:0.11.22 /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY db ./db
RUN uv sync --frozen --no-dev

EXPOSE 8000

# The port is read from the environment at start so one image serves any port.
# Invoking the venv's uvicorn directly (not `uv run`) avoids a network re-sync
# at container start, which would otherwise pull the dev dependency group back
# into the image and undo the --no-dev build.
CMD ["sh", "-c", "exec /app/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
