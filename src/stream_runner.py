"""Plain-text agent runner for SSE streaming via subprocess.

Monkey-patches trace() to print ALL output to stdout (instead of stderr) with
extra verbose detail (full SPARQL queries from data dict, etc.).
Output is parseable by cli_stream_parser.py.
"""

import asyncio
import logging
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Monkey-patch trace() BEFORE any other imports that use it
# ---------------------------------------------------------------------------
import trace as _trace_module

_original_trace = _trace_module.trace


def _stream_trace(
    file: str, msg: str, event_type: str = "log", data: dict | None = None
) -> None:
    """Patched trace: prints to stdout with extra verbose detail."""
    print(f"  \u26a1 [{file}] {msg}", flush=True)

    # For SPARQL events, also print the full query text
    if event_type in ("sparql", "sparql_result") and data and "query" in data:
        q = data["query"]
        print(f"  \u26a1 [{file}] SPARQL: {q}", flush=True)

    # For tool_execution events, print collected args in detail
    if event_type == "tool_execution" and data and "args" in data:
        args = data["args"]
        for k, v in args.items():
            vs = str(v)
            if len(vs) > 500:
                vs = vs[:500] + "..."
            print(f"  \u26a1 [{file}]   Arg {k}: {vs}", flush=True)

    # Don't call original — it prints to stderr which would cause duplicates
    # when subprocess stderr is merged into stdout. The TraceCollector path
    # is irrelevant here since there's no SSE queue in the subprocess.


_trace_module.trace = _stream_trace

# ---------------------------------------------------------------------------
# Now safe to import agent modules
# ---------------------------------------------------------------------------
from config import settings
from flavors import create_agent_session
from agent.agent import AgentResponseType


def _print_header(query: str) -> None:
    print(f"Query: {query}", flush=True)
    print(f"LLM: {settings.llm.ollama.model_name}", flush=True)
    print(f"Flavor: {settings.flavor.description()}", flush=True)
    print(f"---", flush=True)


async def main() -> None:
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What products does Amazon sell?"
    _print_header(query)

    console_start = datetime.now()

    session = create_agent_session(
        after_query_parser_callback=lambda ctx, dur, tok: None,
        after_reasoning_step_callback=lambda resp, dur, tok: None,
        after_tool_callback=lambda tc, to, dur: print(
            f"Executed tool '{tc.tool_name}'. {dur:.4f}s", flush=True
        ),
    )

    answer = await session.generate_answer(query=query, top_n=10, debug=True)
    duration = (datetime.now() - console_start).total_seconds()

    if answer and answer.raw_answer and "maximum number of steps" in answer.raw_answer.lower():
        print(f"Answer: Agent was terminated because it reached the maximum number of steps!", flush=True)
    elif answer:
        desc = answer.augmented_answer.description if answer.augmented_answer else ""
        print(f"Answer: {desc}", flush=True)
        entities = (
            answer.augmented_answer.links
            if answer.augmented_answer and answer.augmented_answer.links
            else answer.entities
        )
        for entity in entities:
            types = ", ".join(entity.types)
            print(f"  [{entity.id}] {entity.name} ({types})", flush=True)

    print(f"Total duration: {duration:.2f}s", flush=True)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.ERROR)
    asyncio.run(main())
