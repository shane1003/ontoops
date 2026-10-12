"""LLM utilities: prompt loading, JSON parsing (light repair), completion with parse retry.

Shared by experts (and future LLM callers). An LLM sometimes emits invalid JSON (trailing
commas, mixed-in explanations etc.), so `complete_json` re-calls with a correction
instruction on a parse failure and tries a light repair; only if that still fails does the
caller degrade.

(The previously duplicated and cross-imported `load_prompt`/`parse_json_object` are
consolidated here.)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from app.core.config import Settings

log = logging.getLogger("conceptualizer.llm.util")


def load_prompt(settings: Settings, filename: str, **fmt: str) -> str:
    """Loads a prompt from config/prompts/ (plain substitution of {key} placeholders only).

    str.format would break the literal JSON braces in the prompts, so only {key} is
    substituted.
    """
    text = (Path(settings.paths.prompts_dir) / filename).read_text(encoding="utf-8")
    for key, value in fmt.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        inner = text.split("```", 2)
        text = inner[1] if len(inner) >= 2 else text
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    return text


def _outer_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _repair(text: str) -> str:
    """Light JSON repair: removes trailing commas (,} / ,]).

    (Used for the last attempt after a parse failure.)
    """
    return re.sub(r",(\s*[}\]])", r"\1", text)


def parse_json_object(text: str) -> dict:
    """Extracts and parses the outermost JSON object of an LLM response.

    Code fences and stray text are tolerated. If the first parse fails, it retries once
    after a light repair (trailing commas removed).
    """
    body = _outer_object(_strip_fences(text))
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return json.loads(_repair(body))  # retry after repair (raises if it still fails)


def complete_json(
    client,
    system: str,
    user: str,
    *,
    label: str,
    fast: bool = True,
    attempts: int = 3,
) -> dict:
    """Gets a JSON response via client.complete and parses it.

    On a parse failure it re-calls with a correction instruction. The LLM occasionally
    emits invalid JSON, so it retries up to `attempts` times (the retry prompt appends the
    instruction "the previous response was not valid JSON ..."). If the last attempt
    fails too, the last exception is raised so the caller degrades.
    """
    last_exc: Exception | None = None
    prompt = user
    for i in range(1, attempts + 1):
        text = client.complete(system, prompt, fast=fast, label=label)
        try:
            return parse_json_object(text)
        except Exception as exc:  # noqa: BLE001 — a parse failure is absorbed by the retry
            last_exc = exc
            log.warning(
                "%s: JSON parse failed (attempt %d/%d) → retrying: %s", label, i, attempts, exc
            )
            prompt = (
                user
                + "\n\n[중요] 직전 응답이 유효한 JSON 이 아니었다. 설명·코드펜스 없이 "
                "**유효한 JSON 객체 하나만** 반환하라. 후행 콤마 금지, 문자열 내 "
                "따옴표는 이스케이프하라."
            )
    assert last_exc is not None
    raise last_exc
