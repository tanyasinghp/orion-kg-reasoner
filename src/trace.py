"""Simple execution tracer: call trace(file, msg) at key points."""

import sys


def trace(file: str, msg: str) -> None:
    """Print a trace message prefixed with the source filename."""
    print(f"  ⚡ [{file}] {msg}", file=sys.stderr)
