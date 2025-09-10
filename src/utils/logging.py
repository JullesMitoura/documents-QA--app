import logging
import sys

# Predefined log formats
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
DETAILED_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | [%(name)s:%(lineno)d - %(funcName)s] | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logger(
    name: str = __name__,
    level: int = logging.INFO,
    detailed: bool = False,
    stream=sys.stdout,
) -> logging.Logger:
    """
    Setup a logger with consistent formatting.

    Args:
        name (str): Logger name (usually __name__).
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        detailed (bool): If True, include filename, line and function info.
        stream: Stream for log output (default: sys.stdout).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent log duplication if called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(stream)
        formatter = logging.Formatter(
            fmt=DETAILED_LOG_FORMAT if detailed else DEFAULT_LOG_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT,
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Optional: avoid propagating logs to root logger
        logger.propagate = False

    return logger