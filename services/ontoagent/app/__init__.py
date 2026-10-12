"""Multi-module app package: assembles the per-module routers into one FastAPI server.

- `app/<module>/` : one module (router, schemas, service logic). Currently: conceptualizer.
- `app/core/`     : infrastructure shared by the modules (config loader, LLM client/tracing).
- `app/main.py`   : app assembly (router includes) and the server entry point.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
