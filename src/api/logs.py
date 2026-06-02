import logging
from typing import Any

from config import settings


def log_config() -> dict[str, Any]:
    """Get the logging configuration for the web service providing the API to interact with the agent.

    This function yields a `Python logging configuration`_ in the configuration dictionary schema.

    Returns:
        A dictionary containing the configuration for the loggers used by the API service.

    .. _Python logging configuration:
        https://docs.python.org/3/library/logging.config.html
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "logging.Formatter",
                "fmt": "%(asctime)s - %(levelname)s - %(message)s",
            },
            "uvicorn-access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": "%(asctime)s - %(levelname)s - %(request_line)s %(status_code)s",
            }
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout"
            },
            "uvicorn-access": {
                "class": "logging.StreamHandler",
                "formatter": "uvicorn-access",
                "stream": "ext://sys.stdout"
            }
        },
        "loggers": {
            "api": {
                "level": settings.api.log_level,
                "handlers": ["default"],
                "propagate": False
            },
            "agent": {
                "level": settings.api.log_level,
                "handlers": ["default"],
                "propagate": False
            },
            "tools": {
                "level": settings.api.log_level,
                "handlers": ["default"],
                "propagate": False
            },
            "uvicorn.access": {
                "level": logging.getLevelName(logging.INFO),
                "handlers": ["uvicorn-access"],
                "propagate": False
            },
            "uvicorn.error": {
                "level": logging.getLevelName(logging.INFO),
                "handlers": ["default"],
                "propagate": False
            }
        },
        "root": {
            "level": logging.getLevelName(logging.CRITICAL),
            "handlers": ["default"]
        }
    }
