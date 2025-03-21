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


def create_parser():
    """Create and configure the argument parser with all supported commands."""
    # flake8: noqa
    balancr_ascii = """
  ____        _                       
 | __ )  __ _| | __ _ _ __   ___ _ __ 
 |  _ \\ / _` | |/ _` | '_ \\ / __| '__|
 | |_) | (_| | | (_| | | | | (__| |   
 |____/ \\__,_|_|\\__,_|_| |_|\\___|_|   
                                     
"""

    # Create the main parser
    parser = argparse.ArgumentParser(
        prog="balancr",
        description=f"{balancr_ascii}\nA command-line tool for analysing and comparing techniques for handling imbalanced datasets.",
        epilog="""
Getting Started:
  1. Load your data:            e.g. balancr load-data your_file.csv -t target_column
  2. Preprocess data:           e.g. balancr preprocess --scale standard --handle-missing mean
  3. Select Techniques:         e.g. balancr select-techniques SMOTE ADASYN
  4. Select Classifier          e.g. balancr select-classifier RandomForest
  5. Configure Metrics          e.g. balancr configure-metrics --metrics precision recall --save-formats csv
  6. Configure Visualisations   e.g. balancr configure-visualisations --types all --save-formats png pdf
  7. Configure Evaluation       e.g. balancr configure-evaluation --test-size 0.3 --cross-validation 5
  8. Run comparison!            e.g. balancr run

  You can also make more efficient and direct configurations via: ~/.balancr/config.json
        
Examples:
  # Load a dataset and examine its class distribution
  balancr load-data data.csv -t target_column

  # Select balancing techniques to compare
  balancr select-techniques SMOTE RandomUnderSampler

  # Run a comparison using current configuration
  balancr run --output-dir results

  # Show all available techniques
  balancr select-techniques --list-available

Full documentation available at: https://github.com/Ruaskill/balancr
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add global options
    parser.add_argument(
        "--version",
        action="version",
        version=f"balancr v{__version__}",
        help="Show the version number and exit",
    )

    # Mutually exclusive group for logging options
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with detailed logging information",
    )
    log_group.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output - only show warnings and errors",
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
    register_select_classifiers_command(subparsers)
    register_configure_metrics_command(subparsers)
    register_configure_visualisations_command(subparsers)
    register_configure_evaluation_command(subparsers)
    register_run_command(subparsers)
    register_reset_command(subparsers)

    return parser


def register_load_data_command(subparsers):
    """Register the load-data command."""
    parser = subparsers.add_parser(
        "load-data",
        help="Load a dataset for analysis",
        description="Load a dataset from a file and configure it for analysis with balancing techniques.",
        epilog="""
Examples:
  # Load a dataset with all features
  balancr load-data dataset.csv -t target

  # Load a dataset with specific features
  balancr load-data dataset.csv -t target -f feature1 feature2 feature3

  # Load from an Excel file
  balancr load-data data.xlsx -t class
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the data file (currently supports CSV, Excel)",
    )
    parser.add_argument(
        "--target-column",
        "-t",
        required=True,
        help="Name of the target/class column in the dataset",
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
        "preprocess",
        help="Configure preprocessing options for the dataset",
        description="Set options for handling missing values, scaling features, and encoding categorical variables.",
        epilog="""
Examples:
  # Configure standard scaling and mean imputation
  balancr preprocess --scale standard --handle-missing mean

  # Skip scaling but encode categorical features
  balancr preprocess --scale none --encode label

  # Remove rows with missing values
  balancr preprocess --handle-missing drop
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--handle-missing",
        choices=["drop", "mean", "median", "mode", "none"],
        default="mean",
        help="How to handle missing values: 'drop' removes rows, 'mean'/'median'/'mode' impute values, 'none' leaves them as-is",
    )
    parser.add_argument(
        "--scale",
        choices=["standard", "minmax", "robust", "none"],
        default="standard",
        help="Scaling method: 'standard' (z-score), 'minmax' (0-1 range), 'robust' (median-based), 'none' (no scaling)",
    )
    parser.add_argument(
        "--encode",
        choices=["auto", "onehot", "label", "ordinal", "none"],
        default="auto",
        help="Encoding for categorical features: 'auto' (detect and convert), 'onehot' (one-hot encoding), 'label' (integer labels), 'none' (no encoding)",
    )
    parser.set_defaults(func=commands.preprocess)


def register_select_techniques_command(subparsers):
    """Register the select-techniques command."""
    parser = subparsers.add_parser(
        "select-techniques",
        help="Select balancing techniques to compare",
        description="Specify which data balancing techniques to use in the comparison.",
        epilog="""
Examples:
  # View all available techniques
  balancr select-techniques --list-available

  # Select single technique
  balancr select-techniques SMOTE

  # Select multiple techniques for comparison
  balancr select-techniques ADASYN BorderlineSMOTE SMOTETomek
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "techniques",
        nargs="*",
        help="Names of balancing techniques to compare (use --list-available to see options)",
        default=[],
    )

    group.add_argument(
        "-l",
        "--list-available",
        action="store_true",
        help="List all available balancing techniques",
    )
    parser.set_defaults(func=commands.select_techniques)


def register_select_classifiers_command(subparsers):
    """Register the select-classifiers command."""
    parser = subparsers.add_parser(
        "select-classifiers",
        help="Select classifier(s) for evaluation",
        description="Choose which classification algorithm(s) to use when evaluating balanced datasets.",
        epilog="""
Examples:
  # Use Random Forest with default settings (replaces existing classifiers)
  balancr select-classifiers RandomForestClassifier

  # Select multiple classifiers
  balancr select-classifiers RandomForestClassifier LogisticRegression SVC

  # Add classifiers without replacing existing ones
  balancr select-classifiers -a LogisticRegression

  # List all available classifiers
  balancr select-classifiers --list-available
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument(
        "classifiers",
        nargs="*",
        help="Names of classifiers to use (use --list-available to see options)",
        default=[],
    )
    
    group.add_argument(
        "-l",
        "--list-available",
        action="store_true",
        help="List all available classifiers",
    )
    
    parser.add_argument(
        "-a",
        "--append",
        action="store_true",
        help="Add to existing classifiers instead of replacing them",
    )
    
    parser.set_defaults(func=commands.select_classifier)


def register_configure_metrics_command(subparsers):
    """Register the configure-metrics command."""
    parser = subparsers.add_parser(
        "configure-metrics",
        help="Configure metrics for evaluation",
        description="Specify which performance metrics to use when comparing balancing techniques.",
        epilog="""
Examples:
  # Use the default set of metrics
  balancr configure-metrics

  # Use only precision and recall
  balancr configure-metrics --metrics precision recall

  # Save results in both CSV and JSON formats
  balancr configure-metrics --save-formats csv json
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["precision", "recall", "f1", "roc_auc"],
        help="Metrics to use for evaluation (default: precision, recall, f1, roc_auc). Options include: accuracy, precision, recall, f1, roc_auc, specificity, g_mean, average_precision",
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
        "configure-visualisations",
        help="Configure visualisation options",
        description="Set options for generating and displaying visual comparisons of balancing techniques.",
        epilog="""
Examples:
  # Generate all visualisation types
  balancr configure-visualisations --types all

  # Only generate distribution visualisations
  balancr configure-visualisations --types distribution

  # Save visualisations in multiple formats
  balancr configure-visualisations --save-formats png pdf

  # Display visualisations on screen during execution
  balancr configure-visualisations --display
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["metrics", "distribution", "learning_curves", "all", "none"],
        default=["all"],
        help="Types of visualisations to generate: 'metrics' (performance comparison), 'distribution' (class balance), 'learning_curves' (model performance vs. training size), 'all', or 'none'",
    )
    parser.add_argument(
    "--display",
    dest="display",
    action="store_true",
    help="Display visualisations on screen during execution",
    )
    parser.add_argument(
        "--no-display",
        dest="display",
        action="store_false",
        help="Don't display visualisations during execution",
    )
    parser.set_defaults(display=False)
    parser.add_argument(
        "--save-formats",
        nargs="+",
        choices=["png", "pdf", "svg", "none"],
        default=["png"],
        help="File formats for saving visualisations (default: png)",
    )
    parser.set_defaults(func=commands.configure_visualisations)


def register_configure_evaluation_command(subparsers):
    """Register the configure-evaluation command."""
    parser = subparsers.add_parser(
        "configure-evaluation",
        help="Configure model evaluation settings",
        description="Set options for model training, testing, and evaluation.",
        epilog="""
Examples:
  # Use 30% of data for testing
  balancr configure-evaluation --test-size 0.3

  # Enable 5-fold cross-validation
  balancr configure-evaluation --cross-validation 5

  # Set a specific random seed for reproducibility
  balancr configure-evaluation --random-state 123
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of dataset to use for testing (default: 0.2, range: 0.1-0.5)",
    )
    parser.add_argument(
        "--cross-validation",
        type=int,
        default=0,
        help="Number of cross-validation folds (0 disables cross-validation, recommended range: 3-10)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.set_defaults(func=commands.configure_evaluation)


def register_run_command(subparsers):
    """Register the run command."""
    parser = subparsers.add_parser(
        "run",
        help="Run comparison of balancing techniques",
        description="Execute the comparison of selected balancing techniques using the configured settings.",
        epilog="""
Examples:
  # Run with default output directory
  balancr run

  # Save results to a specific directory
  balancr run --output-dir results/experiment1

  # Full pipeline example:
  #   balancr load-data data.csv -t class
  #   balancr select-techniques SMOTE RandomUnderSampler
  #   balancr configure-metrics --metrics precision recall f1
  #   balancr run --output-dir results
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default="./balancr_results",
        help="Directory to save results (will be created if it doesn't exist)",
    )
    parser.set_defaults(func=commands.run_comparison)


def register_reset_command(subparsers):
    """Register the reset command."""
    parser = subparsers.add_parser(
        "reset",
        help="Reset the configuration to defaults",
        description="Reset all configuration settings to their default values.",
        epilog="""
Examples:
  # Reset all settings to defaults
  balancr reset
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.set_defaults(func=commands.reset_config)


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Determine logging level based on arguments
    if args.verbose:
        log_level = "verbose"
    elif args.quiet:
        log_level = "quiet"
    else:
        log_level = "default"

    # Configure logging
    utils.setup_logging(log_level)

    # Ensure config directory exists
    config_path = Path(args.config_path)
    config_dir = config_path.parent
    config_dir.mkdir(parents=True, exist_ok=True)

    # Initialise configuration if needed
    if not config_path.exists():
        config.initialise_config(config_path)

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
