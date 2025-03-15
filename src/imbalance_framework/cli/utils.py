"""
utils.py - Utility functions for the balancr CLI.

This module provides helper functions for logging, file handling,
and other common tasks used across the CLI application.
"""

import logging
import sys


# Define a filter to exclude the matplotlib 'findfont' messages
class FontMessageFilter(logging.Filter):
    """Filter to remove noisy font-related log messages."""

    def filter(self, record):
        """Filter out 'findfont' log messages from matplotlib."""
        return "findfont" not in record.getMessage()


def setup_logging(log_level="default"):
    """
    Set up logging with colour and the specified verbosity level.

    Args:
        log_level: "verbose", "default", or "quiet"
    """
    # Reset any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    level_map = {
        "verbose": logging.DEBUG,
        "default": logging.INFO,
        "quiet": logging.WARNING,
    }

    # Get the log level, defaulting to INFO
    log_level_value = level_map.get(log_level, logging.INFO)

    # Root logger level
    root_logger.setLevel(log_level_value)

    # Coloured logging
    try:
        import colorama

        colorama.init()

        # Define colour codes
        COLORS = {
            "DEBUG": colorama.Fore.BLUE,
            "INFO": colorama.Fore.GREEN,
            "WARNING": colorama.Fore.YELLOW,
            "ERROR": colorama.Fore.RED,
            "CRITICAL": colorama.Fore.RED + colorama.Style.BRIGHT,
        }

        # Create custom formatter with colours
        class ColoredFormatter(logging.Formatter):
            def format(self, record):
                levelname = record.levelname
                if levelname in COLORS:
                    levelname_color = (
                        COLORS[levelname] + levelname + colorama.Style.RESET_ALL
                    )
                    record.levelname = levelname_color
                return super().format(record)

        # Create console handler with the custom formatter
        console = logging.StreamHandler(sys.stdout)
        formatter = ColoredFormatter("%(levelname)s: %(message)s")
        console.setFormatter(formatter)

        # Add filter for font messages
        console.addFilter(FontMessageFilter())

        root_logger.addHandler(console)

    except ImportError:
        # Fall back to standard logging if colorama is not available
        logging.basicConfig(
            level=log_level_value,
            format="%(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        # Add filter for font messages to the root logger
        for handler in root_logger.handlers:
            handler.addFilter(FontMessageFilter())

    # Set higher levels for some third-party libraries to reduce noise
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # Log the configured level
    if log_level == "verbose":
        logging.debug("Verbose logging enabled")
    elif log_level == "quiet":
        logging.warning("Quiet logging mode - only showing warnings and errors")
