from __future__ import annotations

import logging
from warnings import warn


def get_logger(name: str) -> logging.Logger:
    warn(
        '"get_logger" will be deprected soon.\
         Please use "get_script_logger" instead.',
        DeprecationWarning,
    )
    return logging.getLogger(name)


def get_script_logger(name: str = "") -> logging.Logger:
    return logging.getLogger(f"script.{name}")
