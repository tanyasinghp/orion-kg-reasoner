"""Structured trace events for the UI visualizer."""

import json
import asyncio
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

    def to_sse(self) -> dict[str, Any]:
        """Translate internal trace event to frontend SSE schema."""
        sse_type = self._sse_event_type()
        sse_data: dict[str, Any] = {"type": sse_type}

        if sse_type == "parse_complete":
            d = self.data
            sse_data["intents"] = d.get("intents", [])
            sse_data["target_types"] = d.get("target_types", [])
            sse_data["relevant_types"] = d.get("relevant_types", [])
            sse_data["background_info"] = d.get("background_info_ids", [])
        elif sse_type == "tool_start":
            d = self.data
            sse_data["step"] = self.step
            sse_data["tool_name"] = d.get("tool_name", "")
            sse_data["arguments"] = d.get("args", {})
            sse_data["reason"] = d.get("reason", "")
        elif sse_type == "tool_complete":
            d = self.data
            sse_data["step"] = self.step
            sse_data["tool_name"] = d.get("tool_name", "")
            sse_data["entity_count"] = d.get("entities_count", 0)
            sse_data["relation_count"] = d.get("relations_count", 0)
            sse_data["entities"] = d.get("entities", [])
            sse_data["latency_seconds"] = d.get("latency", 0)
        elif sse_type == "ranker":
            d = self.data
            sse_data["step"] = self.step
            sse_data["before"] = d.get("before", 0)
            sse_data["after"] = d.get("after", 0)
            sse_data["target"] = d.get("target", "entities")
        elif sse_type == "answer_ready":
            d = self.data
            sse_data["answer_text"] = d.get("answer_text", "")
            sse_data["augmented_entities"] = d.get("entities", [])
        elif sse_type == "max_steps_reached":
            pass
        elif sse_type == "step":
            sse_data["step"] = self.data.get("step", self.step)
            sse_data["duration"] = self.data.get("duration", 0)
            sse_data["status"] = self.data.get("status", "")
        elif sse_type == "error":
            sse_data["message"] = self.data.get("error", self.message)

        return sse_data

    def _sse_event_type(self) -> str:
        """Map internal event_type to SSE event type."""
        et = self.event_type
        phase = self.phase
        data = self.data or {}

        if et == "phase":
            if data.get("phase") == "query_parser" and data.get("intents") is not None:
                return "parse_complete"
        if et == "tool_call":
            return "tool_start"
        if et == "tool_result":
            return "tool_complete"
        if et == "ranker":
            return "ranker"
        if et == "step" and data.get("status") == "answer":
            return "answer_ready"
        if et == "answer":
            return "answer_ready"
        if et == "error":
            return "error"
        if et == "step":
            return "step"
        return "log"


class TraceCollector:
    """Collects trace events for UI playback.
    
    Usage:
        collector = TraceCollector()
        collector.emit("step", "agent_loop", 1, "Invoking LLM...")
        events = collector.get_events()  # returns list of dicts
    """
    
    def __init__(self, sse_queue: asyncio.Queue | None = None):
        self._events: list[TraceEvent] = []
        self._current_step = 0
        self._sse_queue = sse_queue

    def emit(self, event_type: str, phase: str, message: str, data: dict | None = None) -> None:
        step = self._current_step if phase != "query_parser" else 0
        event = TraceEvent(event_type, phase, step, message, data)
        self._events.append(event)
        if self._sse_queue is not None:
            sse_data = event.to_sse()
            # Skip unknown/log events for SSE
            if sse_data.get("type") and sse_data["type"] != "log":
                try:
                    self._sse_queue.put_nowait(sse_data)
                except asyncio.QueueFull:
                    pass

    def emit_sse(self, sse_event: dict[str, Any]) -> None:
        """Emit a pre-built SSE event directly (not from internal trace)."""
        if self._sse_queue is not None:
            try:
                self._sse_queue.put_nowait(sse_event)
            except asyncio.QueueFull:
                pass

    def set_step(self, step: int) -> None:
        self._current_step = step

    def get_events(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._events]

    def clear(self) -> None:
        self._events.clear()
        self._current_step = 0
