"""Structured logging for the monitoring Lambda.

Goals
-----
* One JSON object per log line, so CloudWatch Logs Insights can filter and count.
* A correlation id (``invocation_id``) on **every** line of a single run.
* Never log secrets: only identifiers, numbers and short messages.
* Works both inside Lambda (stdout is captured automatically) and locally.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextvars import ContextVar, Token
from typing import Any

# default=None (not {}) so every line that forgets to bind a context still gets a
# clean payload instead of a shared dict another request could mutate.
_LOG_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar("lab_log_context", default=None)

#: Attributes LogRecord adds itself; anything else is part of our payload.
_RESERVED_RECORD_ATTRS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
        "levelname", "levelno", "lineno", "message", "module", "msecs", "msg", "name",
        "pathname", "process", "processName", "relativeCreated", "stack_info",
        "thread", "threadName", "taskName",
    }
)


class JsonFormatter(logging.Formatter):
    """Render each record as a compact JSON document."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.") + f"{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        payload.update(_LOG_CONTEXT.get() or {})

        event = getattr(record, "event", None)
        if event:
            payload["event"] = event

        fields = getattr(record, "event_fields", None)
        if fields:
            payload["fields"] = _json_safe(fields)

        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_ATTRS and not key.startswith("_") and key not in payload:
                payload[key] = _json_safe(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, separators=(",", ":"))


def _json_safe(value: Any) -> Any:
    """Best-effort conversion of a value into something ``json.dumps`` accepts."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def configure_logging(level: str | int | None = None) -> None:
    """Install the JSON formatter on the root logger (idempotent)."""
    resolved = level or os.environ.get("LOG_LEVEL", "INFO")
    numeric = getattr(logging, str(resolved).upper(), logging.INFO)
    if not isinstance(numeric, int):
        numeric = logging.INFO

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(numeric)

    # CrewAI and boto3 are chatty at DEBUG level; keep them at WARNING unless the
    # operator explicitly raises LOG_LEVEL.
    for noisy in ("botocore", "boto3", "urllib3", "crewai", "chromadb", "opentelemetry"):
        logging.getLogger(noisy).setLevel(max(numeric, logging.WARNING))


def get_logger(name: str) -> logging.Logger:
    """Return a logger whose ``name`` ends up in the JSON payload."""
    if not name.startswith("lab."):
        name = f"lab.{name}"
    return logging.getLogger(name)


def set_log_context(**fields: Any) -> Token[dict[str, Any] | None]:
    """Attach fields to every subsequent log line (used for the invocation id)."""
    current = dict(_LOG_CONTEXT.get() or {})
    current.update({key: value for key, value in fields.items() if value is not None})
    return _LOG_CONTEXT.set(current)


def reset_log_context(token: Token[dict[str, Any] | None]) -> None:
    """Undo a previous :func:`set_log_context`."""
    _LOG_CONTEXT.reset(token)


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Log a structured event: ``log_event(log, logging.INFO, "sns_published", ...)``."""
    logger.log(level, event, extra={"event": event, "event_fields": fields})


def current_context() -> dict[str, Any]:
    """Return the fields currently attached to the log context."""
    return dict(_LOG_CONTEXT.get() or {})
