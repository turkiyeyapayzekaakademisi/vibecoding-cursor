"""Logging infrastructure for the diabetes prediction project."""

import logging
from pathlib import Path

LOG_FILE_NAME = "diabetes_model.log"


def setup_logging(log_dir: Path | None = None) -> logging.Logger:
    """Configure root logging once and return the project logger."""
    project_root = Path(__file__).resolve().parent.parent
    resolved_log_dir = log_dir or project_root
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = resolved_log_dir / LOG_FILE_NAME

    logger = logging.getLogger("diabetes_tahmini")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger
