"""Structured trace events for the UI visualizer."""

import json
from datetime import datetime, timezone
from typing import Any


class TraceEvent:
    """A structured event emitted during agent execution for UI consumption."""
    
    def __init__(self, event_type: str, phase: str, step: int, message: str, data: dict | None = None):
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.event_type = event_type
        self.phase = phase
        self.step = step
        self.message = message
        self.data = data or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "phase": self.phase,
            "step": self.step,
            "message": self.message,
            "data": self.data,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class TraceCollector:
    """Collects trace events for UI playback.
    
    Usage:
        collector = TraceCollector()
        collector.emit("step", "agent_loop", 1, "Invoking LLM...")
        events = collector.get_events()  # returns list of dicts
    """
    
    def __init__(self):
        self._events: list[TraceEvent] = []
        self._current_step = 0

    def emit(self, event_type: str, phase: str, message: str, data: dict | None = None) -> None:
        step = self._current_step if phase != "query_parser" else 0
        event = TraceEvent(event_type, phase, step, message, data)
        self._events.append(event)

    def set_step(self, step: int) -> None:
        self._current_step = step

    def get_events(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._events]

    def clear(self) -> None:
        self._events.clear()
        self._current_step = 0
