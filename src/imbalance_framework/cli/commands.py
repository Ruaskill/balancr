"""
commands.py - Command handlers for the balancr CLI.

This module contains the implementation of all command functions that are
registered in main.py
"""

import logging
import json
from pathlib import Path

from . import config

# Will be used to interact with the core balancing framework
try:
    from imbalance_framework.imbalance_analyser import BalancingFramework
    from imbalance_framework.technique_registry import TechniqueRegistry
except ImportError:
    logging.error(
        "Could not import balancing framework. Ensure it's installed correctly."
    )
    BalancingFramework = None
    TechniqueRegistry = None


def load_data(args):
    """
    Handle the load-data command.

    Args:
        args: Command line arguments from argparse

    Returns:
        int: Exit code
    """
    logging.info(f"Loading data from {args.file_path}")

    # Validate file exists
    if not Path(args.file_path).exists():
        logging.error(f"File not found: {args.file_path}")
        return 1

    # Update configuration with data file settings
    settings = {
        "data_file": args.file_path,
        "target_column": args.target_column,
    }

    if args.feature_columns:
        settings["feature_columns"] = args.feature_columns

    try:
        # Validate that the file can be loaded
        if BalancingFramework is not None:
            # This is just a validation check, not storing the framework instance
            framework = BalancingFramework()
            framework.load_data(
                args.file_path, args.target_column, args.feature_columns
            )

            # Get and display class distribution
            distribution = framework.inspect_class_distribution(plot=False)
            print("\nClass Distribution:")
            for cls, count in distribution.items():
                print(f"  Class {cls}: {count} samples")

            total = sum(distribution.values())
            for cls, count in distribution.items():
                pct = (count / total) * 100
                print(f"  Class {cls}: {pct:.2f}%")

        # Update config with new settings
        config.update_config(args.config_path, settings)
        logging.info(
            f"Data configuration saved: {args.file_path}, target: {args.target_column}"
        )
        return 0

    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return 1


def preprocess(args):
    """
    Handle the preprocess command.

    Args:
        args: Command line arguments from argparse

    Returns:
        int: Exit code
    """
    logging.info("Configuring preprocessing options")

    # Update configuration with preprocessing settings
    settings = {
        "preprocessing": {
            "handle_missing": args.handle_missing,
            "scale": args.scale,
            "encode": args.encode,
        }
    }

    try:
        current_config = config.load_config(args.config_path)

        # Ensure data file is configured
        if "data_file" not in current_config:
            logging.error("No data file configured. Run 'balancr load-data' first.")
            return 1

        # Update config
        config.update_config(args.config_path, settings)
        logging.info("Preprocessing configuration saved")

        # Display the preprocessing settings
        print("\nPreprocessing Configuration:")
        print(f"  Handle Missing Values: {args.handle_missing}")
        print(f"  Feature Scaling: {args.scale}")
        print(f"  Categorical Encoding: {args.encode}")

        return 0

    except Exception as e:
        logging.error(f"Failed to configure preprocessing: {e}")
        return 1


def select_techniques(args):
    """
    Handle the select-techniques command.

    Args:
        args: Command line arguments from argparse

    Returns:
        int: Exit code
    """
    # List available techniques if requested
    if args.list_available and BalancingFramework is not None:
        print("Listing available balancing techniques...")
        try:
            framework = BalancingFramework()
            techniques = framework.list_available_techniques()

            print("\nAvailable Techniques:")

            # Print custom techniques
            if techniques.get("custom"):
                print("\nCustom Techniques:")
                for technique in techniques["custom"]:
                    print(f"  - {technique}")

            # Print imblearn techniques
            if techniques.get("imblearn"):
                print("\nImbalanced-Learn Techniques:")
                for technique in sorted(techniques["imblearn"]):
                    print(f"  - {technique}")

            return 0

        except Exception as e:
            logging.error(f"Failed to list techniques: {e}")
            return 1

    # When not listing but selecting techniques
    logging.info(f"Selecting balancing techniques: {', '.join(args.techniques)}")

    try:
        # Validate techniques exist if framework is available
        if BalancingFramework is not None:
            framework = BalancingFramework()
            available = framework.list_available_techniques()
            all_techniques = available.get("custom", []) + available.get("imblearn", [])

            invalid_techniques = [t for t in args.techniques if t not in all_techniques]

            if invalid_techniques:
                logging.error(f"Invalid techniques: {', '.join(invalid_techniques)}")
                logging.info(
                    "Use 'balancr select-techniques --list-available' to see available techniques"
                )
                return 1

        # Update configuration with selected techniques
        settings = {"balancers": args.techniques}

        config.update_config(args.config_path, settings)
        logging.info(f"Selected balancing techniques: {', '.join(args.techniques)}")
        return 0

    except Exception as e:
        logging.error(f"Failed to select techniques: {e}")
        return 1


def select_classifier(args):
    """
    Handle the select-classifier command.

    Args:
        args: Command line arguments from argparse

    Returns:
        int: Exit code
    """
    logging.info(f"Selecting classifier: {args.classifier}")

    # Parse classifier parameters if provided
    classifier_params = {}
    if args.params:
        try:
            classifier_params = json.loads(args.params)
        except json.JSONDecodeError:
            logging.error("Invalid JSON for classifier parameters")
            return 1

    # Update configuration with classifier settings
    settings = {"classifier": {"name": args.classifier, "params": classifier_params}}

    try:
        config.update_config(args.config_path, settings)

        # Display confirmation
        print(f"\nSelected classifier: {args.classifier}")
        if classifier_params:
            print("Classifier parameters:")
            for param, value in classifier_params.items():
                print(f"  - {param}: {value}")

        return 0

    except Exception as e:
        logging.error(f"Failed to select classifier: {e}")
        return 1


def configure_metrics(args):
    """
    Handle the configure-metrics command.

    Args:
        args: Command line arguments from argparse

    Returns:
        int: Exit code
    """
    logging.info(f"Configuring metrics: {', '.join(args.metrics)}")

    # Update configuration with metrics settings
    settings = {
        "output": {"metrics": args.metrics, "save_metrics_formats": args.save_formats}
    }

    try:
        # Update existing output settings if they exist
        current_config = config.load_config(args.config_path)
        if "output" in current_config:
            current_output = current_config["output"]
            # Merge with existing output settings without overwriting other output options
            settings["output"] = {**current_output, **settings["output"]}

        config.update_config(args.config_path, settings)

        # Display confirmation
        print("\nMetrics Configuration:")
        print(f"  Metrics: {', '.join(args.metrics)}")
        print(f"  Save Formats: {', '.join(args.save_formats)}")

        return 0

    except Exception as e:
        logging.error(f"Failed to configure metrics: {e}")
        return 1


def configure_visualisations(args):
    """
    Handle the configure-visualisations command.

    Args:
        args: Command line arguments from argparse

    Returns:
        int: Exit code
    """
    types_str = "all visualisations" if "all" in args.types else ", ".join(args.types)
    logging.info(f"Configuring visualisations: {types_str}")

    # Update configuration with visualisation settings
    settings = {
        "output": {
            "visualisations": args.types,
            "display_visualisations": args.display,
            "save_vis_formats": args.save_formats,
        }
    }

    try:
        # Update existing output settings if they exist
        current_config = config.load_config(args.config_path)
        if "output" in current_config:
            current_output = current_config["output"]
            # Merge with existing output settings without overwriting other output options
            settings["output"] = {**current_output, **settings["output"]}

        config.update_config(args.config_path, settings)

        # Display confirmation
        print("\nVisualisation Configuration:")
        print(f"  Types: {types_str}")
        print(f"  Display During Execution: {'Yes' if args.display else 'No'}")
        print(f"  Save Formats: {', '.join(args.save_formats)}")

        return 0

    except Exception as e:
        logging.error(f"Failed to configure visualisations: {e}")
        return 1


def configure_evaluation(args):
    """
    Handle the configure-evaluation command.

    Args:
        args: Command line arguments from argparse

    Returns:
        int: Exit code
    """
    logging.info("Configuring evaluation settings")

    # Update configuration with evaluation settings
    settings = {
        "evaluation": {
            "test_size": args.test_size,
            "cross_validation": args.cross_validation,
            "random_state": args.random_state,
        }
    }

    try:
        config.update_config(args.config_path, settings)

        # Display confirmation
        print("\nEvaluation Configuration:")
        print(f"  Test Size: {args.test_size}")
        print(f"  Cross-Validation Folds: {args.cross_validation}")
        print(f"  Random State: {args.random_state}")

        return 0

    except Exception as e:
        logging.error(f"Failed to configure evaluation: {e}")
        return 1


def run_comparison(args):
    """
    Handle the run command.

    Args:
        args: Command line arguments from argparse

    Returns:
        int: Exit code
    """
    # Load current configuration
    try:
        current_config = config.load_config(args.config_path)
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        return 1

    # Check if all required settings are configured
    required_settings = ["data_file", "target_column", "balancers"]
    missing_settings = [s for s in required_settings if s not in current_config]

    if missing_settings:
        logging.error(f"Missing required configuration: {', '.join(missing_settings)}")
        logging.info("Please configure all required settings before running comparison")
        return 1

    # Ensure balancing framework is available
    if BalancingFramework is None:
        logging.error("Balancing framework not available. Please check installation")
        return 1

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get output and evaluation settings with defaults
    output_config = current_config.get("output", {})
    metrics = output_config.get("metrics", ["precision", "recall", "f1", "roc_auc"])
    visualisations = output_config.get("visualisations", ["all"])
    display_visualisations = output_config.get("display_visualisations", False)
    save_metrics_formats = output_config.get("save_metrics_formats", ["csv"])
    save_vis_formats = output_config.get("save_vis_formats", ["png"])

    eval_config = current_config.get("evaluation", {})
    test_size = eval_config.get("test_size", 0.2)
    # cross_validation = eval_config.get("cross_validation", 0) Will add later
    random_state = eval_config.get("random_state", 42)

    logging.info(
        f"Running comparison with techniques: {', '.join(current_config['balancers'])}"
    )
    logging.info(f"Results will be saved to: {output_dir}")

    try:
        # Initialize the framework
        framework = BalancingFramework()

        # Load data
        logging.info(f"Loading data from {current_config['data_file']}")
        feature_columns = current_config.get("feature_columns", None)
        framework.load_data(
            current_config["data_file"],
            current_config["target_column"],
            feature_columns,
        )

        # Apply preprocessing if configured
        if "preprocessing" in current_config:
            logging.info("Applying preprocessing")
            preproc = current_config["preprocessing"]

            framework.preprocess_data(
                handle_missing=preproc.get("handle_missing", "mean"),
                scale=preproc.get("scale", "standard"),
                encode=preproc.get("encode", "auto"),
            )

        # Determine if visualisations should be plotted during comparison
        plot_during_compare = display_visualisations and "metrics" in visualisations
        if "all" in visualisations:
            plot_during_compare = display_visualisations

        # Run comparison
        logging.info("Comparing balancing techniques")
        results = framework.compare_techniques(
            current_config["balancers"],
            test_size=test_size,
            random_state=random_state,
            plot_results=plot_during_compare,
        )

        # Save metrics in requested formats
        for format_type in save_metrics_formats:
            if format_type != "none":
                results_file = output_dir / f"comparison_results.{format_type}"
                logging.info(f"Saving results to {results_file}")
                framework.save_results(
                    results_file,
                    file_type=format_type,
                    include_plots=False,  # Handle plots separately below
                )

        # Generate and save visualisations
        vis_types_to_generate = []
        if "all" in visualisations:
            vis_types_to_generate = ["metrics", "distribution", "learning_curves"]
        else:
            vis_types_to_generate = visualisations

        for vis_type in vis_types_to_generate:
            if vis_type == "none":
                continue

            for format_type in save_vis_formats:
                if format_type == "none":
                    continue

                if vis_type == "metrics":
                    logging.info(
                        f"Generating metrics comparison plot in {format_type} format"
                    )
                    plot_path = output_dir / f"metrics_comparison.{format_type}"
                    if (
                        not plot_during_compare
                    ):  # Only generate if not already done during comparison
                        framework.plot_comparison_results(
                            results, save_path=str(plot_path)
                        )

                elif vis_type == "distribution":
                    logging.info(
                        f"Generating imbalanced and balanced class distribution plots in {format_type} format"
                    )
                    imbalanced_plot_path = output_dir / f"imbalanced_class_distribution_comparison.{format_type}"
                    framework.inspect_class_distribution(save_path=str(imbalanced_plot_path))
                    balanced_plot_path = output_dir / f"balanced_class_distribution_comparison.{format_type}"
                    framework.compare_balanced_class_distributions(
                        save_path=str(balanced_plot_path)
                    )

                elif vis_type == "learning_curves":
                    logging.info(f"Generating learning curves in {format_type} format")
                    plot_path = output_dir / f"learning_curves.{format_type}"
                    framework.generate_learning_curves(save_path=str(plot_path))

        # Print summary of results
        print("\nResults Summary:")
        for technique, technique_metrics in results.items():
            print(f"\n{technique}:")
            for metric_name, value in technique_metrics.items():
                if metric_name in metrics:
                    print(f"  {metric_name}: {value:.4f}")

        print(f"\nDetailed results saved to: {output_dir}")
        return 0

    except Exception as e:
        logging.error(f"Error during comparison: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def reset_config(args):
    """
    Handle the reset command.

    Args:
        args: Command line arguments from argparse

    Returns:
        int: Exit code
    """
    try:
        config.initialize_config(args.config_path, force=True)
        logging.info("Configuration has been reset to defaults")
        return 0
    except Exception as e:
        logging.error(f"Failed to reset configuration: {e}")
        return 1
