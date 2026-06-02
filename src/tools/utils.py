from typing import Any
from datetime import datetime


def format_value_for_sparql(value: Any) -> str: # noqa: ANN401
    """Format the given value for use in a SPARQL query.

    Args:
        value: Value

    Returns:
        The given value formatted for use in a SPARQL query.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, int):
        return str(value)
    elif isinstance(value, float):
        return str(value)
    elif isinstance(value, str):
        # TODO: Improve date/time parsing logic
        # Try to parse as date
        try:
            _ = datetime.strptime(value, "%Y-%m-%d")  # ISO 8601 date
            return f'"{value}"^^xsd:date'
        except ValueError:
            pass

        # Try to parse as datetime
        for fmt in [
            "%Y-%m-%dT%H:%M:%S",  # ISO 8601 datetime
            "%Y-%m-%dT%H:%M:%SZ",  # ISO 8601 datetime with timezone
        ]:
            try:
                _ = datetime.strptime(value, fmt)
                return f'"{value}"^^xsd:dateTime'
            except ValueError:
                pass
        # normal string
        return f'"{value}"'
    else:
        raise ValueError(f"Unsupported type: {type(value)}")
