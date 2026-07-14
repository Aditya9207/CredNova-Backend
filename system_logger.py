"""
Centralized system logging for CredNova backend.

Logs to console and logs/crednova.log with a consistent format.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_FILE = _LOG_DIR / "crednova.log"
_CONFIGURED = False

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, level_name, logging.INFO)

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("crednova")
    root.setLevel(log_level)
    root.handlers.clear()
    root.propagate = False

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(component: str) -> logging.Logger:
    """Return a namespaced logger, e.g. crednova.ml, crednova.http."""
    setup_logging()
    return logging.getLogger(f"crednova.{component}")


def log_path() -> Path:
    """Absolute path to the active log file."""
    setup_logging()
    return _LOG_FILE
