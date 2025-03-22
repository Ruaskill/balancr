"""
commands.py - Command handlers for the balancr CLI.

This module contains the implementation of all command functions that are
registered in main.py
"""

import logging
import json
import inspect
import numpy as np
from pathlib import Path

from evaluation.visualisation import plot_comparison_results

from . import config

# Will be used to interact with the core balancing framework
try:
    from imbalance_framework.imbalance_analyser import BalancingFramework
    from imbalance_framework.technique_registry import TechniqueRegistry
    from imbalance_framework.classifier_registry import ClassifierRegistry
except ImportError as e:
    logging.error(f"Could not import balancing framework: {str(e)}")
    logging.error(
        "Could not import balancing framework. Ensure it's installed correctly."
    )
    BalancingFramework = None
    TechniqueRegistry = None
    ClassifierRegistry = None


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
            distribution = framework.inspect_class_distribution(display=False)
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
    # Check if we should list available classifiers
    if args.list_available:
        return list_available_classifiers(args)

    logging.info(f"Selecting classifiers: {', '.join(args.classifiers)}")

    # Create classifier registry
    if ClassifierRegistry is None:
        logging.error("Classifier registry not available. Please check installation.")
        return 1

    registry = ClassifierRegistry()

    # Get classifier configurations
    classifier_configs = {}

    for classifier_name in args.classifiers:
        # Get the classifier class
        classifier_class = registry.get_classifier_class(classifier_name)

        if classifier_class is None:
            logging.error(f"Classifier '{classifier_name}' not found.")
            logging.info(
                "Use 'balancr select-classifier --list-available' to see available classifiers."
            )
            continue

        # Get default parameters
        params = get_classifier_default_params(classifier_class)
        classifier_configs[classifier_name] = params

    # If no valid classifiers were found
    if not classifier_configs:
        logging.error("No valid classifiers selected.")
        return 1

    try:
        # Read existing config (we need this regardless of append mode)
        current_config = config.load_config(args.config_path)

        if args.append:
            # Append mode: Update existing classifiers
            existing_classifiers = current_config.get("classifiers", {})
            existing_classifiers.update(classifier_configs)
            settings = {"classifiers": existing_classifiers}

            print(f"\nAdded classifiers: {', '.join(classifier_configs.keys())}")
            print(f"Total classifiers: {', '.join(existing_classifiers.keys())}")
        else:
            # Replace mode: Create a completely new config entry
            # We'll create a copy of the current config and explicitly set the classifiers
            new_config = dict(current_config)  # shallow copy is sufficient
            new_config["classifiers"] = classifier_configs

            # Use config.write_config instead of update_config to replace the entire file
            config_path = Path(args.config_path)
            with open(config_path, "w") as f:
                json.dump(new_config, f, indent=2)

            print(
                f"\nReplaced classifiers with: {', '.join(classifier_configs.keys())}"
            )

            # Return early since we've manually written the config
            print("Default parameters have been added to the configuration file.")
            print("You can modify them by editing the configuration or using the CLI.")
            return 0

        # Only reach here in append mode
        config.update_config(args.config_path, settings)

        print("Default parameters have been added to the configuration file.")
        print("You can modify them by editing the configuration or using the CLI.")

        return 0
    except Exception as e:
        logging.error(f"Failed to select classifiers: {e}")
        return 1


def list_available_classifiers(args):
    """
    List all available classifiers.

    Args:
        args: Command line arguments from argparse

    Returns:
        int: Exit code
    """
    if ClassifierRegistry is None:
        logging.error("Classifier registry not available. Please check installation.")
        return 1

    registry = ClassifierRegistry()
    classifiers = registry.list_available_classifiers()

    print("\nAvailable Classifiers:")

    # Print sklearn classifiers by module
    if "sklearn" in classifiers:
        print("\nScikit-learn Classifiers:")
        for module_name, clf_list in classifiers["sklearn"].items():
            print(f"\n  {module_name.capitalize()}:")
            for clf in sorted(clf_list):
                print(f"    - {clf}")

    # Print custom classifiers if any
    if "custom" in classifiers and classifiers["custom"]:
        print("\nCustom Classifiers:")
        for module_name, clf_list in classifiers["custom"].items():
            if clf_list:
                print(f"\n  {module_name.capitalize()}:")
                for clf in sorted(clf_list):
                    print(f"    - {clf}")

    return 0


def get_classifier_default_params(classifier_class):
    """
    Extract default parameters from a classifier class.

    Args:
        classifier_class: The classifier class to inspect

    Returns:
        Dictionary of parameter names and their default values
    """
    params = {}

    try:
        # Get the signature of the __init__ method
        sig = inspect.signature(classifier_class.__init__)

        # Process each parameter
        for name, param in sig.parameters.items():
            # Skip 'self' parameter
            if name == "self":
                continue

            # Get default value if it exists
            if param.default is not inspect.Parameter.empty:
                # Handle special case for None (JSON uses null)
                if param.default is None:
                    params[name] = None
                # Handle other types that can be serialised to JSON
                elif isinstance(param.default, (int, float, str, bool, list, dict)):
                    params[name] = param.default
                else:
                    # Convert non-JSON-serialisable defaults to string representation
                    params[name] = str(param.default)
            else:
                # For parameters without defaults, use None
                params[name] = None

    except Exception as e:
        logging.warning(
            f"Error extracting parameters from {classifier_class.__name__}: {e}"
        )

    return params


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
            "learning_curve_folds": args.learning_curve_folds,
            "learning_curve_points": args.learning_curve_points
        }
    }

    try:
        config.update_config(args.config_path, settings)

        # Display confirmation
        print("\nEvaluation Configuration:")
        print(f"  Test Size: {args.test_size}")
        print(f"  Cross-Validation Folds: {args.cross_validation}")
        print(f"  Random State: {args.random_state}")
        print(f"  Learning Curve Folds: {args.learning_curve_folds}")
        print(f"  Learning Curve Points: {args.learning_curve_points}")

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
    cv_enabled = eval_config.get("cross_validation", 0) > 0
    cv_folds = eval_config.get("cross_validation", 5)
    random_state = eval_config.get("random_state", 42)

    logging.info(
        f"Running comparison with techniques: {', '.join(current_config['balancers'])}"
    )
    logging.info(f"Results will be saved to: {output_dir}")

    try:
        # Initialise the framework
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
            logging.info("Applying preprocessing...")
            preproc = current_config["preprocessing"]

            handle_missing = preproc.get("handle_missing", "mean")
            scale = preproc.get("scale", "standard")
            encode = preproc.get("encode", "auto")

            framework.preprocess_data(
                handle_missing=handle_missing,
                scale=scale,
                encode=encode,
            )
            logging.info("Data preprocessing applied")

        # Apply balancing techniques
        logging.info("Applying balancing techniques...")
        framework.apply_balancing_techniques(
            current_config["balancers"],
            test_size=test_size,
            random_state=random_state,
        )
        logging.info("Balancing techniques applied successfully")

        # Save balanced datasets at the root level
        balanced_dir = output_dir / "balanced_datasets"
        balanced_dir.mkdir(exist_ok=True)
        logging.info(f"Saving balanced datasets to {balanced_dir}")
        framework.generate_balanced_data(
            folder_path=str(balanced_dir),
            techniques=current_config["balancers"],
            file_format="csv"
        )

        # Train classifiers
        logging.info("Training classifiers on balanced datasets...")
        classifiers = current_config.get("classifiers", {})
        if not classifiers:
            logging.warning(
                "No classifiers configured. Using default RandomForestClassifier."
            )

        # Train classifiers with the balanced datasets
        results = framework.train_classifiers(
            classifier_configs=classifiers, enable_cv=cv_enabled, cv_folds=cv_folds
        )

        logging.info("Training and evaluation complete")

        # Determine which visualisation types to generate
        vis_types_to_generate = []
        if "all" in visualisations:
            vis_types_to_generate = ["metrics", "distribution", "learning_curves"]
        else:
            vis_types_to_generate = visualisations

        # Save class distribution visualisations at the root level
        for format_type in save_vis_formats:
            if format_type == "none":
                continue

            if "distribution" in vis_types_to_generate or "all" in visualisations:
                # Original (imbalanced) class distribution
                logging.info(
                    f"Generating imbalanced class distribution in {format_type} format"
                )
                imbalanced_plot_path = (
                    output_dir / f"imbalanced_class_distribution.{format_type}"
                )
                framework.inspect_class_distribution(
                    save_path=str(imbalanced_plot_path), display=display_visualisations
                )

                # Balanced class distributions comparison
                logging.info(
                    f"Generating balanced class distribution comparison in {format_type} format"
                )
                balanced_plot_path = (
                    output_dir / f"balanced_class_distribution.{format_type}"
                )
                framework.compare_balanced_class_distributions(
                    save_path=str(balanced_plot_path),
                    display=display_visualisations,
                )

        # Process each classifier and save its results in a separate directory
        for classifier_name in current_config.get("classifiers", {}):
            logging.info(f"Processing results for classifier: {classifier_name}")

            # Create classifier-specific directory
            classifier_dir = output_dir / classifier_name
            classifier_dir.mkdir(exist_ok=True)

            # Create standard metrics directory
            std_metrics_dir = classifier_dir / "standard_metrics"
            std_metrics_dir.mkdir(exist_ok=True)

            # Save standard metrics in requested formats
            for format_type in save_metrics_formats:
                if format_type == "none":
                    continue

                results_file = std_metrics_dir / f"comparison_results.{format_type}"
                logging.info(
                    f"Saving standard metrics for {classifier_name} to {results_file}"
                )

                # We need a modified save_results method that can extract a specific classifier's results
                framework.save_classifier_results(
                    results_file,
                    classifier_name=classifier_name,
                    metric_type="standard_metrics",
                    file_type=format_type,
                )

            # Generate and save standard metrics visualisations
            for format_type in save_vis_formats:
                if format_type == "none":
                    continue

                print(f"vis_types_to_generate: '{vis_types_to_generate}'")
                if "metrics" in vis_types_to_generate or "all" in visualisations:
                    metrics_path = std_metrics_dir / f"metrics_comparison.{format_type}"
                    logging.info(
                        f"Generating metrics comparison for {classifier_name} in {format_type} format"
                    )

                    # Call a modified plot_comparison_results that can handle specific classifier data
                    plot_comparison_results(
                        results,
                        classifier_name=classifier_name,
                        metric_type="standard_metrics",
                        save_path=str(metrics_path),
                        display=display_visualisations,
                    )

                if "learning_curves" in vis_types_to_generate or "all" in visualisations:
                    learning_curve_path = (
                        std_metrics_dir / f"learning_curves.{format_type}"
                    )
                    logging.info(
                        f"Generating learning curves for {classifier_name} in {format_type} format"
                    )

                    # Get learning curve parameters from config
                    learning_curve_points = eval_config.get("learning_curve_points", 10)
                    learning_curve_folds = eval_config.get("learning_curve_folds", 5)
                    train_sizes = np.linspace(0.1, 1.0, learning_curve_points)

                    # Generate learning curves for particular classifier
                    framework.generate_learning_curves(
                        classifier_name=classifier_name,
                        train_sizes=train_sizes,
                        n_folds=learning_curve_folds,
                        save_path=str(learning_curve_path),
                        display=display_visualisations
                    )

            # If cross-validation is enabled, create CV metrics directory and save results
            if cv_enabled:
                cv_metrics_dir = classifier_dir / "cv_metrics"
                cv_metrics_dir.mkdir(exist_ok=True)

                # Save CV metrics in requested formats
                for format_type in save_metrics_formats:
                    if format_type == "none":
                        continue

                    cv_results_file = (
                        cv_metrics_dir / f"comparison_results.{format_type}"
                    )
                    logging.info(
                        f"Saving CV metrics for {classifier_name} to {cv_results_file}"
                    )

                    framework.save_classifier_results(
                        cv_results_file,
                        classifier_name=classifier_name,
                        metric_type="cv_metrics",
                        file_type=format_type,
                    )

                # Generate and save CV metrics visualisations
                for format_type in save_vis_formats:
                    if format_type == "none":
                        continue

                    if "metrics" in vis_types_to_generate or "all" in visualisations:
                        cv_metrics_path = (
                            cv_metrics_dir / f"metrics_comparison.{format_type}"
                        )
                        logging.info(
                            f"Generating CV metrics comparison for {classifier_name} in {format_type} format"
                        )

                        plot_comparison_results(
                            results,
                            classifier_name=classifier_name,
                            metric_type="cv_metrics",
                            save_path=str(cv_metrics_path),
                            display=display_visualisations,
                        )

                    if (
                        "learning_curves" in vis_types_to_generate
                        or "all" in visualisations
                    ):
                        cv_learning_curve_path = (
                            cv_metrics_dir / f"learning_curves.{format_type}"
                        )
                        logging.info(
                            f"Generating CV learning curves for {classifier_name} in {format_type} format"
                        )

                        # Get learning curve parameters from config
                        learning_curve_points = eval_config.get("learning_curve_points", 10)
                        learning_curve_folds = eval_config.get("learning_curve_folds", 5)
                        train_sizes = np.linspace(0.1, 1.0, learning_curve_points)

                        framework.generate_learning_curves(
                            classifier_name=classifier_name,
                            train_sizes=train_sizes,
                            n_folds=learning_curve_folds,
                            save_path=str(cv_learning_curve_path),
                            display=display_visualisations
                        )

        # Print summary of results
        print("\nResults Summary:")
        for classifier_name, classifier_results in results.items():
            print(f"\n{classifier_name}:")
            for technique_name, technique_metrics in classifier_results.items():
                print(f"  {technique_name}:")
                if "standard_metrics" in technique_metrics:
                    std_metrics = technique_metrics["standard_metrics"]
                    for metric_name, value in std_metrics.items():
                        if metric_name in metrics:
                            print(f"    {metric_name}: {value:.4f}")

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
        config.initialise_config(args.config_path, force=True)
        logging.info("Configuration has been reset to defaults")
        return 0
    except Exception as e:
        logging.error(f"Failed to reset configuration: {e}")
        return 1
