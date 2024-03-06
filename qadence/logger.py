from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    # TODO, add deprecation warning
    return logging.getLogger(name)


def get_script_logger(name: str = "") -> logging.Logger:
    return logging.getLogger(f"script.{name}")
