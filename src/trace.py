"""Simple execution tracer: call trace(file, msg) at key points.

Supports optional TraceCollector for structured events consumed by the UI.
Uses contextvars to keep collectors per-async-task (prevents SSE cross-talk).
"""

import sys
import contextvars
from trace_event import TraceCollector


_collector_var: contextvars.ContextVar[TraceCollector | None] = contextvars.ContextVar(
    "trace_collector", default=None
)


def set_collector(collector: TraceCollector | None) -> None:
    _collector_var.set(collector)


def get_collector() -> TraceCollector | None:
    return _collector_var.get()


def trace(file: str, msg: str, event_type: str = "log", data: dict | None = None) -> None:
    """Print a trace message and optionally emit a structured event."""
    print(f"  \u26a1 [{file}] {msg}", file=sys.stderr)
    collector = _collector_var.get()
    if collector is not None:
        collector.emit(event_type, file, msg, data)
