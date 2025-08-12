"""Simple logging setup for the churn project."""

import logging
import os
from pathlib import Path

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
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Console handler (stdout)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        # Create log directory if it doesn't exist
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


DEFAULT_LOGGER = setup_logging()
