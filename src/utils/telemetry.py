"""Minimal telemetry helpers: JSON logging, trace_id, and optional spans."""

from __future__ import annotations

import contextvars
import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional
from uuid import uuid4

try:  # Optional OpenTelemetry support
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    _otel_available = True
except Exception:  # pragma: no cover
    _otel_available = False

_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)


def get_trace_id() -> Optional[str]:
    return _trace_id.get()


def set_trace_id(trace_id: str) -> None:
    _trace_id.set(trace_id)


def new_trace_id() -> str:
    tid = uuid4().hex
    set_trace_id(tid)
    return tid


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: Dict[str, Any] = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
        }
        trace_id = getattr(record, "trace_id", None) or get_trace_id()
        if trace_id:
            payload["trace_id"] = trace_id
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "WARNING") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.WARNING))
    formatter = JsonFormatter()
    root = logging.getLogger()
    for handler in root.handlers:
        handler.setFormatter(formatter)


def configure_otel(service_name: str = "mcp-sim") -> None:
    if not _otel_available:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)


@contextmanager
def span(name: str, attrs: Optional[Dict[str, Any]] = None):
    """
    Lightweight span wrapper. If OpenTelemetry is available, emits a real span,
    otherwise just measures duration and logs at DEBUG.
    """
    logger = logging.getLogger(__name__)
    start = time.perf_counter()
    otel_span = None
    if _otel_available:
        tracer = trace.get_tracer(__name__)
        otel_span = tracer.start_span(name=name, attributes=attrs or {})
        otel_cm = trace.use_span(otel_span, end_on_exit=True)
        otel_cm.__enter__()
    try:
        yield
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 3)
        logger.debug(json.dumps({"event": "span", "name": name, "duration_ms": duration_ms, "attrs": attrs or {}}, ensure_ascii=False))
        if otel_span:
            otel_cm.__exit__(None, None, None)
