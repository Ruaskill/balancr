#!/usr/bin/env python3
"""
main.py - Entry point for the balancr CLI.

This module sets up the command-line interface for the balancr framework,
which provides tools for comparing different data balancing techniques.
"""
import argparse
import sys
import logging
from pathlib import Path

# Import commands module (will be implemented next)
from . import commands
from . import config
from . import utils

# CLI version
__version__ = "0.1.0"


def setup_logging(verbose):
    """Configure logging based on verbosity level."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    utils.setup_colored_logging()


def create_parser():
    """Create and configure the argument parser with all supported commands."""
    # Create the main parser
    parser = argparse.ArgumentParser(
        prog="balancr",
        description="A command-line tool for analysing and comparing data balancing techniques.",
        epilog="Run 'balancr COMMAND --help' for more information on a specific command.",
    )

    # Add global options
    parser.add_argument(
        "--version", action="version", version=f"balancr v{__version__}"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--config-path",
        default=Path.home() / ".balancr" / "config.json",
        help="Path to the configuration file (default: ~/.balancr/config.json)",
    )

    # Create subparsers for each command
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Register all commands
    register_load_data_command(subparsers)
    register_preprocess_command(subparsers)
    register_select_techniques_command(subparsers)
    register_select_classifier_command(subparsers)
    register_configure_metrics_command(subparsers)
    register_configure_visualisations_command(subparsers)
    register_configure_evaluation_command(subparsers)
    register_run_command(subparsers)
    register_reset_command(subparsers)

    return parser


def register_load_data_command(subparsers):
    """Register the load-data command."""
    parser = subparsers.add_parser("load-data", help="Load a dataset for analysis")
    parser.add_argument(
        "file_path", type=str, help="Path to the data file (CSV, Excel, etc.)"
    )
    parser.add_argument(
        "--target-column",
        "-t",
        required=True,
        help="Name of the target column in the dataset",
    )
    parser.add_argument(
        "--feature-columns",
        "-f",
        nargs="+",
        help="Names of feature columns to use (default: all except target)",
    )
    parser.set_defaults(func=commands.load_data)


def register_preprocess_command(subparsers):
    """Register the preprocess command."""
    parser = subparsers.add_parser(
        "preprocess", help="Configure preprocessing options for the dataset"
    )
    parser.add_argument(
        "--handle-missing",
        choices=["drop", "mean", "median", "mode", "none"],
        default="mean",
        help="How to handle missing values in the dataset",
    )
    parser.add_argument(
        "--scale",
        choices=["standard", "minmax", "robust", "none"],
        default="standard",
        help="Scaling method to apply to features",
    )
    parser.add_argument(
        "--encode",
        choices=["auto", "onehot", "label", "ordinal", "none"],
        default="auto",
        help="Encoding method for categorical features",
    )
    parser.set_defaults(func=commands.preprocess)


def register_select_techniques_command(subparsers):
    """Register the select-techniques command."""
    parser = subparsers.add_parser(
        "select-techniques", help="Select balancing techniques to compare"
    )
    parser.add_argument(
        "techniques", nargs="+", help="Names of balancing techniques to use"
    )
    parser.add_argument(
        "--list-available",
        action="store_true",
        help="List all available balancing techniques",
    )
    parser.set_defaults(func=commands.select_techniques)


def register_select_classifier_command(subparsers):
    """Register the select-classifier command."""
    parser = subparsers.add_parser(
        "select-classifier", help="Select classifier for evaluation"
    )
    parser.add_argument(
        "classifier",
        choices=["RandomForest", "LogisticRegression", "SVM", "KNN", "DecisionTree"],
        help="Classifier to use for evaluating balancing techniques",
    )
    parser.add_argument(
        "--params", type=str, help="JSON string with classifier parameters"
    )
    parser.set_defaults(func=commands.select_classifier)


def register_configure_metrics_command(subparsers):
    """Register the configure-metrics command."""
    parser = subparsers.add_parser(
        "configure-metrics", help="Configure metrics for evaluation"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["precision", "recall", "f1", "roc_auc"],
        help="Metrics to use for evaluation",
    )
    parser.add_argument(
        "--save-formats",
        nargs="+",
        choices=["csv", "json", "none"],
        default=["csv"],
        help="Formats to save metrics data (default: csv)",
    )
    parser.set_defaults(func=commands.configure_metrics)


def register_configure_visualisations_command(subparsers):
    """Register the configure-visualisations command."""
    parser = subparsers.add_parser(
        "configure-visualisations", help="Configure visualisation options"
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["metrics", "distribution", "learning_curves", "all", "none"],
        default=["all"],
        help="Visualisations to generate",
    )
    parser.add_argument(
        "--display", action="store_true", help="Display visualisations during execution"
    )
    parser.add_argument(
        "--save-formats",
        nargs="+",
        choices=["png", "pdf", "svg", "none"],
        default=["png"],
        help="Formats to save visualisations (default: png)",
    )
    parser.set_defaults(func=commands.configure_visualisations)


def register_configure_evaluation_command(subparsers):
    """Register the configure-evaluation command."""
    parser = subparsers.add_parser(
        "configure-evaluation", help="Configure model evaluation settings"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of dataset to use for testing",
    )
    parser.add_argument(
        "--cross-validation",
        type=int,
        default=0,
        help="Number of cross-validation folds (0 disables cross-validation)",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.set_defaults(func=commands.configure_evaluation)


def register_run_command(subparsers):
    """Register the run command."""
    parser = subparsers.add_parser("run", help="Run comparison of balancing techniques")
    parser.add_argument(
        "--output-dir",
        default="./balancr_results",
        help="Directory to save results (default: ./balancr_results)",
    )
    parser.set_defaults(func=commands.run_comparison)


def register_reset_command(subparsers):
    """Register the reset command."""
    parser = subparsers.add_parser("reset", help="Reset the configuration to defaults")
    parser.set_defaults(func=commands.reset_config)


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Configure logging
    setup_logging(args.verbose)

    # Ensure config directory exists
    config_path = Path(args.config_path)
    config_dir = config_path.parent
    config_dir.mkdir(parents=True, exist_ok=True)

    # Initialize configuration if needed
    if not config_path.exists():
        config.initialize_config(config_path)

    # If no command is provided, print help
    if not args.command:
        parser.print_help()
        return 0

    try:
        # Call the appropriate command function
        return args.func(args)
    except Exception as e:
        logging.error(f"{e}")
        if args.verbose:
            logging.exception("Detailed traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
