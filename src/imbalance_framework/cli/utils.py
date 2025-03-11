"""
utils.py - Utility functions for the balancr CLI.

This module provides helper functions for logging, file handling,
and other common tasks used across the CLI application.
"""

import logging
import sys


def setup_colored_logging():
    """
    Set up colored logging for better console output.
    Requires the 'colorama' package to be installed.
    """
    try:
        import colorama

        colorama.init()

        # Define color codes
        COLORS = {
            "DEBUG": colorama.Fore.BLUE,
            "INFO": colorama.Fore.GREEN,
            "WARNING": colorama.Fore.YELLOW,
            "ERROR": colorama.Fore.RED,
            "CRITICAL": colorama.Fore.RED + colorama.Style.BRIGHT,
        }

        # Create a custom formatter with colors
        class ColoredFormatter(logging.Formatter):
            def format(self, record):
                levelname = record.levelname
                if levelname in COLORS:
                    levelname_color = (
                        COLORS[levelname] + levelname + colorama.Style.RESET_ALL
                    )
                    record.levelname = levelname_color
                return super().format(record)

        # Get the root logger
        root_logger = logging.getLogger()

        # Remove any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create console handler with the custom formatter
        console = logging.StreamHandler(sys.stdout)
        formatter = ColoredFormatter("%(levelname)s: %(message)s")
        console.setFormatter(formatter)
        root_logger.addHandler(console)

    except ImportError:
        # Fall back to standard logging if colorama is not available
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
