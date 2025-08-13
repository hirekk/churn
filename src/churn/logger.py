"""Simple logging setup for the churn project."""

import logging
import os
from pathlib import Path
import sys

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def setup_logging(
    level: str = LOG_LEVEL,
    log_file: str | None = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """Set up logging with console and optional file output.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file
        log_format: Log message format

    Returns:
        Configured logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    logger.handlers.clear()

    formatter = logging.Formatter(log_format)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


DEFAULT_LOGGER = setup_logging()
