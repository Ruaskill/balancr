"""
config.py - Configuration management for the balancr CLI.

This module handles loading, saving, and updating the configuration
file that stores user settings between command executions.
"""

import json
import logging
from pathlib import Path


def initialise_config(config_path: str, force: bool = False) -> None:
    """
    Initialise a new configuration file with default settings.

    Args:
        config_path: Path to the configuration file
        force: Whether to overwrite an existing file

    Raises:
        IOError: If the file exists and force is False
    """
    config_path = Path(config_path)

    # Check if file already exists
    if config_path.exists() and not force:
        logging.info(f"Configuration file already exists at {config_path}")
        return

    # Create default configuration
    default_config = {
        "preprocessing": {
            "handle_missing": "mean",
            "scale": "standard",
            "encode": "auto",
        },
        "evaluation": {
            "test_size": 0.2,
            "cross_validation": 0,
            "random_state": 42,
            "learning_curve_folds": 5,
            "learning_curve_points": 10
            },
        "output": {
            "metrics": ["precision", "recall", "f1", "roc_auc"],
            "visualisations": ["all"],
            "display_visualisations": False,
            "save_metrics_formats": ["csv"],
            "save_vis_formats": ["png"],
        },
    }

    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Write configuration file
    with open(config_path, "w") as f:
        json.dump(default_config, f, indent=2)

    logging.info(f"Initialised configuration file at {config_path}")


def load_config(config_path: str) -> dict:
    """
    Load configuration from file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def update_config(config_path: str, settings: dict) -> None:
    """
    Update existing configuration with new settings.

    Args:
        config_path: Path to the configuration file
        settings: Dictionary containing settings to update

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_path = Path(config_path)

    # Load existing config
    try:
        current_config = load_config(config_path)
    except FileNotFoundError:
        # Initialise if it doesn't exist
        initialise_config(config_path)
        current_config = load_config(config_path)

    # Update configuration recursively
    deep_update(current_config, settings)

    # Write updated configuration
    with open(config_path, "w") as f:
        json.dump(current_config, f, indent=2)

    logging.debug(f"Updated configuration at {config_path}")


def deep_update(original: dict, update: dict) -> dict:
    """
    Recursively update a dictionary.

    Args:
        original: Dictionary to update
        update: Dictionary with updates

    Returns:
        The updated original dictionary
    """
    for key, value in update.items():
        if (
            isinstance(value, dict)
            and key in original
            and isinstance(original[key], dict)
        ):
            # Recursively update nested dictionaries
            deep_update(original[key], value)
        else:
            # Update or add values
            original[key] = value

    return original


def print_config(config_path: str) -> int:
    """
    Print the current configuration.

    Args:
        config_path: Path to the configuration file

    Returns:
        Exit code
    """
    try:
        config = load_config(config_path)

        print("\nCurrent Configuration:")
        print(json.dumps(config, indent=2))

        return 0
    except Exception as e:
        logging.error(f"Error reading configuration: {e}")
        return 1
