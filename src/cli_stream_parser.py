"""State-machine parser that consumes raw CLI stdout lines and yields structured SSE events.

Maintains a current step context. Every line is accumulated into the raw_lines buffer
for the active context. When a context boundary is crossed, the buffered detail is flushed
into an SSE event dict and emitted.
"""

from __future__ import annotations

import json
import re
from enum import Enum


class ParserState(Enum):
    INIT = 0
    PARSE = 1       # accumulating Phase 1 lines
    IDLE = 2        # between steps (no active tool)
    STEP = 3        # within a tool execution step
    DONE = 4        # terminal — answer or max-steps emitted


# Regex patterns for line classification
_RE_PHASE_1_DONE = re.compile(
    r"Phase 1 done:\s+intents=\[(.*?)\],\s+target_types=\[(.*?)\]"
)
_RE_TOOL_DECISION = re.compile(
    r"Step\s+(\d+):\s+LLM decided TOOL_CALL '([^']+)'"
)
_RE_TOOL_RETURNED = re.compile(
    r"Tool '([^']+)' returned:\s*(\d+)\s+entities,\s*(\d+)\s+relations"
)
_RE_EXECUTED_TOOL = re.compile(
    r"Executed tool '([^']+)'\.\s*([\d.]+)s"
)
_RE_MAX_STEPS = re.compile(r"maximum number of steps", re.IGNORECASE)
_RE_SPARQL_RESULT = re.compile(
    r"Result:\s+(.*)"
)
_RE_RANKER_ENTITIES = re.compile(
    r"rank_and_select_entities:\s*(\d+)\s*→\s*(\d+)\s*\(limit=(\d+)\)"
)
_RE_RANKER_RELATIONS = re.compile(
    r"rank_and_select_relations:\s*(\d+)\s*→\s*(\d+)\s*\(limit=(\d+)\)"
)
_RE_ENTITY_FEATURE = re.compile(
    r"Entity '([^']+)' features=\{(.*)\}"
)
_RE_SCORE_LINE = re.compile(
    r"\[E-\d+\].*score="
)
_RE_URI_LINE = re.compile(
    r"http://amazon/kg/"
)
_RE_REASON = re.compile(
    r"Reason:\s*(.*)"
)
_RE_ARG = re.compile(
    r"Arg\s+(\w+):\s+(.*)"
)
_RE_ANSWER = re.compile(
    r"^Answer:\s*(.*)"
)
_RE_SPARQL_TRACE = re.compile(
    r"SPARQL:\s*(.*)"
)


class CliStreamParser:
    """Read lines from CLI stdout and yield SSE event dicts.

    Usage:
        parser = CliStreamParser()
        for line in lines:
            for event in parser.feed(line):
                yield event
        for event in parser.flush():
            yield event
    """

    def __init__(self) -> None:
        self.state = ParserState.INIT
        self.current_step = 0
        self.tool_name: str | None = None
        self.tool_args: dict | None = None
        self.tool_reason: str | None = None
        self._latency: float = 0.0

        # Buffers
        self._raw_lines: list[str] = []
        self._idle_lines: list[str] = []
        self._sparql_queries: list[str] = []
        self._sparql_results: list[str] = []
        self._entity_features: list[str] = []
        self._entities_in: int = 0
        self._entities_out: int = 0
        self._relations_in: int = 0
        self._relations_out: int = 0
        self._ranker_limit: int | None = None
        self._ranker_scores: list[str] = []
        self._uri_list: list[str] = []

        # Loop detection: {(tool_name, args_json): step_of_first_occurrence}
        self._loop_history: dict[tuple[str, str], int] = {}

        # Answer accumulation
        self._answer_lines: list[str] = []

        # Flag set when "LLM returned ANSWER" line is seen
        self._expecting_answer = False

        # Multi-line SPARQL body tracking
        self._in_sparql_body = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, line: str) -> list[dict]:
        """Feed one raw CLI output line. Returns zero or more SSE event dicts."""
        stripped = line.rstrip("\n\r")
        stripped = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", stripped)
        if not stripped:
            return []
        return self._route_line(stripped)

    def flush(self) -> list[dict]:
        """Flush remaining context. Call after last line to get final events."""
        events: list[dict] = []
        if self.state == ParserState.STEP:
            events.extend(self._emit_tool_complete())
        if self._answer_lines:
            events.extend(self._emit_answer())
        self.state = ParserState.DONE
        return events

    # ------------------------------------------------------------------
    # Line routing
    # ------------------------------------------------------------------

    def _route_line(self, line: str) -> list[dict]:
        events: list[dict] = []

        # ---- Detect transitions ----
        m_phase = _RE_PHASE_1_DONE.search(line)
        m_tool_dec = _RE_TOOL_DECISION.search(line)
        m_exec = _RE_EXECUTED_TOOL.search(line)
        m_answer = _RE_ANSWER.match(line)

        # INIT → PARSE (everything before Phase 1 done)
        if self.state == ParserState.INIT:
            self.state = ParserState.PARSE

        # PARSE: Phase 1 done line flushes the parse event
        if self.state == ParserState.PARSE and m_phase:
            self._accumulate(line)
            events.extend(self._emit_parse_complete())
            self.state = ParserState.IDLE
            return events

        # STEP: Executed tool line flushes the step
        if self.state == ParserState.STEP and m_exec:
            self._latency = float(m_exec.group(2))
            self._accumulate(line)
            events.extend(self._emit_tool_complete())
            self.state = ParserState.IDLE
            return events

        # Detect "LLM returned ANSWER" to flag expecting_answer (before accum)
        _RE_LLM_ANSWER = re.compile(r"Step\s+\d+:\s+LLM returned ANSWER")
        m_llm_ans = _RE_LLM_ANSWER.search(line)
        if m_llm_ans:
            self._expecting_answer = True

        # Any state: Tool decision starts a new step
        if m_tool_dec:
            if self.state == ParserState.STEP:
                events.extend(self._emit_tool_complete())
            self.current_step = int(m_tool_dec.group(1))
            self.tool_name = m_tool_dec.group(2)
            self.tool_args = self._parse_args_from_line(line)
            self.tool_reason = None
            self._reset_step_buffers()

            # Prepend any idle lines accumulated before this tool decision
            if self._idle_lines:
                self._raw_lines = list(self._idle_lines)
                self._idle_lines = []

            self.state = ParserState.STEP
            self._accumulate(line)
            events.extend(self._emit_tool_start())
            return events

        # Answer: line — terminal event
        TERMINATION_STRING = "reached the maximum number of steps"
        if m_answer:
            if self.state == ParserState.STEP:
                events.extend(self._emit_tool_complete())
            answer_text = m_answer.group(1).strip()
            if TERMINATION_STRING in answer_text:
                self.state = ParserState.DONE
                self._raw_lines.append(line)
                events.extend(self._emit_max_steps())
            else:
                self._expecting_answer = False
                self.state = ParserState.DONE
                self._answer_lines.append(line)
                events.extend(self._emit_answer())
            return events

        # Max steps in non-answer line — exact match only
        if line.startswith("Answer:") and TERMINATION_STRING in line:
            self._raw_lines.append(line)
            events.extend(self._emit_max_steps())
            self.state = ParserState.DONE
            return events

        # Accumulate into current state buffer
        self._accumulate(line)
        return events

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def _accumulate(self, line: str) -> None:
        """Add raw line to the active buffer, extracting structured data."""
        # Route to the right buffer
        if self.state == ParserState.IDLE:
            self._idle_lines.append(line)
            return
        elif self.state == ParserState.DONE:
            return

        self._raw_lines.append(line)

        # ---- Multi-line SPARQL body continuation ----
        # Lines without the "  ⚡" trace prefix are continuation of a
        # multi-line SPARQL query printed by stream_runner.py.
        if self._in_sparql_body:
            if not line.startswith("  \u26a1"):
                if self._sparql_queries:
                    self._sparql_queries[-1] += "\n" + line
                return
            self._in_sparql_body = False

        # SPARQL query line — may be followed by a multi-line body
        if _RE_SPARQL_TRACE.search(line):
            self._sparql_queries.append(line)
            self._in_sparql_body = True
            return

        # SPARQL result count
        if m := _RE_SPARQL_RESULT.search(line):
            self._sparql_results.append(m.group(1))

        # Entity feature line
        if _RE_ENTITY_FEATURE.search(line):
            self._entity_features.append(line)

        # Ranker entity pruning
        if m := _RE_RANKER_ENTITIES.search(line):
            self._entities_in = int(m.group(1))
            self._entities_out = int(m.group(2))
            self._ranker_limit = int(m.group(3))

        # Ranker relation pruning
        if m := _RE_RANKER_RELATIONS.search(line):
            self._relations_in = int(m.group(1))
            self._relations_out = int(m.group(2))

        # Score line
        if _RE_SCORE_LINE.search(line):
            self._ranker_scores.append(line)

        # URI line
        if _RE_URI_LINE.search(line):
            self._uri_list.append(line.strip())

        # Tool reason
        if m := _RE_REASON.search(line):
            self.tool_reason = m.group(1)

        # Tool arg
        if m := _RE_ARG.search(line):
            if self.tool_args is None:
                self.tool_args = {}
            self.tool_args[m.group(1)] = m.group(2)

    def _reset_step_buffers(self) -> None:
        self._raw_lines = []
        self._sparql_queries = []
        self._sparql_results = []
        self._entity_features = []
        self._entities_in = 0
        self._entities_out = 0
        self._relations_in = 0
        self._relations_out = 0
        self._ranker_limit = None
        self._ranker_scores = []
        self._uri_list = []
        self._latency = 0.0
        self._in_sparql_body = False

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    def _emit_parse_complete(self) -> list[dict]:
        intents: list[str] = []
        target_types: list[str] = []
        bg_info: list[str] = []
        for line in self._raw_lines:
            if m := _RE_PHASE_1_DONE.search(line):
                intents = [
                    x.strip().strip("'\"")
                    for x in m.group(1).split(",")
                    if x.strip()
                ]
                target_types = [
                    x.strip().strip("'\"")
                    for x in m.group(2).split(",")
                    if x.strip()
                ]
            for m2 in re.finditer(r"\[(B-\d+)\]", line):
                bid = m2.group(1)
                if bid not in bg_info:
                    bg_info.append(bid)

        event: dict = {
            "type": "parse_complete",
            "intents": intents,
            "target_types": target_types,
            "background_info": bg_info,
            "raw_lines": list(self._raw_lines),
        }
        self._raw_lines = []
        return [event]

    def _emit_tool_start(self) -> list[dict]:
        args_hash = json.dumps(self.tool_args or {}, sort_keys=True)
        loop_key = (self.tool_name or "", args_hash)

        is_loop = loop_key in self._loop_history
        if is_loop:
            loop_count = len([
                k for k in self._loop_history
                if k == loop_key
            ]) + 1
        else:
            loop_count = 1
            self._loop_history[loop_key] = self.current_step

        event: dict = {
            "type": "tool_start",
            "step": self.current_step,
            "tool_name": self.tool_name or "",
            "args": self.tool_args or {},
            "reason": self.tool_reason or "",
            "loop": is_loop,
            "loop_count": loop_count,
        }
        return [event]

    def _emit_tool_complete(self) -> list[dict]:
        events: list[dict] = []

        entity_count = 0
        relation_count = 0
        for line in self._raw_lines:
            if m := _RE_TOOL_RETURNED.search(line):
                entity_count = int(m.group(2))
                relation_count = int(m.group(3))

        event: dict = {
            "type": "tool_complete",
            "step": self.current_step,
            "tool_name": self.tool_name or "",
            "entity_count": entity_count,
            "relation_count": relation_count,
            "latency_seconds": self._latency,
            "sparql_queries": list(self._sparql_queries),
            "sparql_results": list(self._sparql_results),
            "uri_list": list(self._uri_list),
            "entity_features": list(self._entity_features),
            "ranker_detail": {
                "entities_in": self._entities_in,
                "entities_out": self._entities_out,
                "limit": self._ranker_limit,
                "scores": list(self._ranker_scores),
                "relations_in": self._relations_in,
                "relations_out": self._relations_out,
            },
            "raw_lines": list(self._raw_lines),
        }
        events.append(event)

        # Separate ranker_fired event if entities were pruned
        if (
            self._entities_in > 0
            and self._entities_out >= 0
            and self._entities_in != self._entities_out
        ):
            events.append({
                "type": "ranker_fired",
                "step": self.current_step,
                "entities_in": self._entities_in,
                "entities_out": self._entities_out,
                "limit": self._ranker_limit,
                "scores": list(self._ranker_scores),
            })

        self._reset_step_buffers()
        return events

    def _emit_answer(self) -> list[dict]:
        answer_text = ""
        for line in self._answer_lines:
            if m := _RE_ANSWER.match(line):
                answer_text = m.group(1).strip()
                if answer_text:
                    break

        event: dict = {
            "type": "answer_ready",
            "answer_text": answer_text,
            "raw_lines": list(self._answer_lines),
        }
        self._answer_lines = []
        return [event]

    def _emit_max_steps(self) -> list[dict]:
        """Build max_steps_reached event."""
        event: dict = {
            "type": "max_steps_reached",
            "raw_lines": list(self._raw_lines),
        }
        self._raw_lines = []
        return [event]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_args_from_line(line: str) -> dict:
        if m := re.search(r"args=\{(.*?)\}", line):
            raw = "{" + m.group(1) + "}"
            try:
                fixed = raw.replace("'", '"')
                return json.loads(fixed)
            except (json.JSONDecodeError, ValueError):
                pass
        return {}
