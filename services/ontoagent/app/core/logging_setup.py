"""Logging setup shared by the modules: keeps the API path no quieter than the CLI.

uvicorn configures only its own loggers and leaves root at WARNING, so doing nothing
makes every pipeline logger.info vanish on the API path, down to the lines that explain a
degrade. The two modules each solved this on their own (old tboxcraft/logging_setup.py +
configure_logging in app/main.py); that is folded into one implementation that takes only
the env variable and the target logger as parameters.

- ``logger_name=None`` (tboxcraft style): targets the **root** logger. If it already has
  handlers, only the level is raised (no duplicate output); otherwise basicConfig.
- ``logger_name="pkg"`` (conceptualizer style): raises only the level of that **package
  logger**. Without touching the root level, record creation is decided by the effective
  level of the emitting logger, so INFO survives and propagates to the ancestor handlers.
  A handler is attached only when nobody has attached one (respects pytest, uvicorn).
"""

from __future__ import annotations

import logging
import os

DEFAULT_LEVEL = "INFO"
_FORMAT = "%(levelname)s %(name)s: %(message)s"


def resolve_level(level: str | int | None = None, *, env_var: str) -> int:
    """Resolves the level from the argument, then the env variable, then the default.

    An unknown name falls back to the default instead of raising: a typo in the
    deployment env must not kill startup, and the warning below makes it visible.
    """
    if isinstance(level, int):
        return level
    raw = (level or os.environ.get(env_var) or DEFAULT_LEVEL).strip()
    resolved = logging.getLevelName(raw.upper())
    if not isinstance(resolved, int):
        logging.getLogger(__name__).warning(
            "unknown %s %r; falling back to %s", env_var, raw, DEFAULT_LEVEL
        )
        return logging.getLevelName(DEFAULT_LEVEL)
    return resolved


def configure_logging(
    level: str | int | None = None,
    *,
    env_var: str,
    logger_name: str | None = None,
    fmt: str = _FORMAT,
    force: bool = False,
) -> int:
    """Configures logging and returns the applied level.

    Idempotent: safe under repeated calls and calls from several modules.
    """
    resolved = resolve_level(level, env_var=env_var)
    root = logging.getLogger()
    if logger_name is None:
        if root.handlers and not force:
            root.setLevel(resolved)
        else:
            logging.basicConfig(level=resolved, format=fmt, force=force)
    else:
        pkg = logging.getLogger(logger_name)
        pkg.setLevel(resolved)
        if not root.handlers and not pkg.handlers:
            logging.basicConfig(level=resolved, format=fmt)
    return resolved
