"""Simple execution tracer: call trace(file, msg) at key points.

Supports optional TraceCollector for structured events consumed by the UI.
"""

import sys
from trace_event import TraceCollector


_collector: TraceCollector | None = None


def set_collector(collector: TraceCollector | None) -> None:
    global _collector
    _collector = collector


def get_collector() -> TraceCollector | None:
    return _collector


def trace(file: str, msg: str, event_type: str = "log", data: dict | None = None) -> None:
    """Print a trace message and optionally emit a structured event."""
    print(f"  ⚡ [{file}] {msg}", file=sys.stderr)
    if _collector is not None:
        _collector.emit(event_type, file, msg, data)
