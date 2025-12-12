"""
Logging configuration for Agentic Reviewer.

Provides structured logging with configurable levels and formats.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    verbose: bool = False,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure and return the root logger for the application.
    
    Args:
        level: Logging level (default: INFO)
        verbose: If True, set level to DEBUG
        log_file: Optional file path to write logs
        
    Returns:
        Configured root logger
    """
    if verbose:
        level = logging.DEBUG
    
    # Create logger
    logger = logging.getLogger("agentic_reviewer")
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with appropriate format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Use different formats based on level
    if level == logging.DEBUG:
        console_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%H:%M:%S"
        )
    else:
        # Minimal format for normal operation (doesn't clutter terminal UI)
        console_format = logging.Formatter("%(message)s")
    
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # Optional file handler with full details
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a specific module.
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"agentic_reviewer.{name}")

