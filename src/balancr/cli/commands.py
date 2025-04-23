"""
commands.py - Command handlers for the balancr CLI.

This module contains the implementation of all command functions that are
registered in main.py
"""

import shutil
import time
from datetime import datetime
import importlib
import logging
import json
import inspect
from balancr.data import DataPreprocessor
import numpy as np
from pathlib import Path

from balancr.evaluation import (
    plot_comparison_results,
    plot_radar_chart,
    plot_3d_scatter,
)
import pandas as pd

from . import config
from balancr import BaseBalancer

# Will be used to interact with the core balancing framework
try:
    from balancr import BalancingFramework
    from balancr import TechniqueRegistry
    from balancr import ClassifierRegistry
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

            # Try to get correlation_threshold from config
            try:
                current_config = config.load_config(args.config_path)
                correlation_threshold = current_config["preprocessing"].get(
                    "correlation_threshold", 0.95
                )
            except Exception:
                # Fall back to default if config can't be loaded or doesn't have the value
                correlation_threshold = 0.95

            print(f"Correlation Threshold: {correlation_threshold}")
            framework.load_data(
                args.file_path,
                args.target_column,
                args.feature_columns,
                correlation_threshold=correlation_threshold,
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

    # Initialise basic preprocessing settings
    settings = {
        "preprocessing": {
            "handle_missing": args.handle_missing,
            "handle_constant_features": args.handle_constant_features,
            "handle_correlations": args.handle_correlations,
            "correlation_threshold": args.correlation_threshold,
            "scale": args.scale,
            "encode": args.encode,
            "save_preprocessed": args.save_preprocessed,
        }
    }

    try:
        current_config = config.load_config(args.config_path)

        # Check if categorical features are specified
        if args.categorical_features:
            # Process categorical features if a dataset is available
            if "data_file" in current_config:
                data_file = current_config["data_file"]
                target_column = current_config.get("target_column")

                logging.info(f"Loading dataset from {data_file} for encoding analysis")

                try:
                    # Initialise framework to load data
                    # Read the dataset directly
                    df = pd.read_csv(data_file)

                    # Validate target column exists
                    if target_column and target_column not in df.columns:
                        logging.warning(
                            f"Target column '{target_column}' not found in dataset"
                        )

                    # Create preprocessor and determine encoding types
                    preprocessor = DataPreprocessor()
                    categorical_encodings = preprocessor.assign_encoding_types(
                        df=df,
                        categorical_columns=args.categorical_features,
                        encoding_type=args.encode,
                        hash_components=args.hash_components,
                        ordinal_columns=args.ordinal_features,
                    )

                    # Add categorical feature encodings to settings
                    settings["preprocessing"][
                        "categorical_features"
                    ] = categorical_encodings

                    # Display encoding recommendations
                    print("\nCategorical feature encoding assignments:")
                    for column, encoding in categorical_encodings.items():
                        print(f"  {column}: {encoding}")
                except Exception as e:
                    logging.error(f"Error analysing dataset for encoding: {e}")
                    logging.info(
                        "Storing categorical features without encoding recommendations"
                    )
                    # If analysis fails, just store the categorical features with the default encoding
                    settings["preprocessing"]["categorical_features"] = {
                        col: args.encode for col in args.categorical_features
                    }
            else:
                logging.warning(
                    "No dataset configured. Cannot analyse categorical features."
                )
                logging.info("Storing categorical features with default encoding")
                # Store categorical features with the specified encoding type
                settings["preprocessing"]["categorical_features"] = {
                    col: args.encode for col in args.categorical_features
                }

        # Update config
        config.update_config(args.config_path, settings)
        logging.info("Preprocessing configuration saved")

        # Display the preprocessing settings
        print("\nPreprocessing Configuration:")
        print(f"  Handle Missing Values: {args.handle_missing}")
        print(f"  Handle Constant Features: {args.handle_constant_features}")
        print(f"  Handle Feature Correlations: {args.handle_correlations}")
        print(f"  Correlation Threshold: {args.correlation_threshold}")
        print(f"  Feature Scaling: {args.scale}")
        print(f"  Default Categorical Encoding: {args.encode}")
        print(f"  Save Preprocessed Data to File: {args.save_preprocessed}")

        if args.categorical_features:
            print(f"  Categorical Features: {', '.join(args.categorical_features)}")

        if args.ordinal_features:
            print(f"  Ordinal Features: {', '.join(args.ordinal_features)}")

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

            # Create technique configurations with default parameters
            balancing_techniques = {}
            for technique_name in args.techniques:
                # Get default parameters for this technique
                params = framework.technique_registry.get_technique_default_params(
                    technique_name
                )
                balancing_techniques[technique_name] = params

        # Read existing config
        current_config = config.load_config(args.config_path)

        if args.append and "balancing_techniques" in current_config:
            # Append mode: Update existing techniques
            existing_techniques = current_config.get("balancing_techniques", {})

            # Add new techniques to the existing ones
            if BalancingFramework is not None:
                existing_techniques.update(balancing_techniques)

            # Update config with merged values
            settings = {
                "balancing_techniques": existing_techniques,
                "include_original_data": args.include_original_data,
            }

            config.update_config(args.config_path, settings)

            print(f"\nAdded balancing techniques: {', '.join(args.techniques)}")
            print(f"Total techniques: {', '.join(existing_techniques.keys())}")
        else:
            # Replace mode: Create a completely new config entry
            # Create a copy of the current config and set the techniques
            new_config = dict(current_config)  # shallow copy
            new_config["balancing_techniques"] = (
                balancing_techniques if BalancingFramework is not None else {}
            )
            new_config["include_original_data"] = args.include_original_data

            # Directly write the entire config to replace the file
            config_path = Path(args.config_path)
            with open(config_path, "w") as f:
                json.dump(new_config, f, indent=2)

            print(f"\nReplaced balancing techniques with: {', '.join(args.techniques)}")

            # Return early since we've manually written the config
            print("Default parameters have been added to the configuration file.")
            print("You can modify them by editing the configuration file.")
            return 0

        print("Default parameters have been added to the configuration file.")
        print("You can modify them by editing the configuration file.")
        return 0

    except Exception as e:
        logging.error(f"Failed to select techniques: {e}")
        return 1


def register_techniques(args):
    """
    Handle the register-techniques command.

    This command allows users to register custom balancing techniques from
    Python files or folders for use in comparisons.

    Args:
        args: Command line arguments from argparse

    Returns:
        int: Exit code
    """
    try:
        # Handle removal operations
        if args.remove or args.remove_all:
            return _remove_techniques(args)

        # Ensure framework is available
        if TechniqueRegistry is None:
            logging.error(
                "Technique registry not available. Please check installation."
            )
            return 1

        registry = TechniqueRegistry()

        # Track successfully registered techniques
        registered_techniques = []

        # Process file path
        if args.file_path:
            file_path = Path(args.file_path)

            if not file_path.exists():
                logging.error(f"File not found: {file_path}")
                return 1

            if not file_path.is_file() or file_path.suffix.lower() != ".py":
                logging.error(f"Not a Python file: {file_path}")
                return 1

            # Register techniques from the file
            registered = _register_from_file(
                registry, file_path, args.name, args.class_name, args.overwrite
            )
            registered_techniques.extend(registered)

        # Process folder path
        elif args.folder_path:
            folder_path = Path(args.folder_path)

            if not folder_path.exists():
                logging.error(f"Folder not found: {folder_path}")
                return 1

            if not folder_path.is_dir():
                logging.error(f"Not a directory: {folder_path}")
                return 1

            # Register techniques from all Python files in the folder
            for py_file in folder_path.glob("*.py"):
                registered = _register_from_file(
                    registry,
                    py_file,
                    None,  # Don't use custom name for folder scanning
                    None,  # Don't use class name for folder scanning
                    args.overwrite,
                )
                registered_techniques.extend(registered)

        # Print summary
        if registered_techniques:
            print("\nSuccessfully registered techniques:")
            for technique in registered_techniques:
                print(f"  - {technique}")

            # Suggestion for next steps
            print("\nYou can now use these techniques in comparisons. For example:")
            print(f"  balancr select-techniques {registered_techniques[0]}")
            return 0
        else:
            logging.warning("No valid balancing techniques found to register.")
            return 1

    except Exception as e:
        logging.error(f"Error registering techniques: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def _register_from_file(
    registry, file_path, custom_name=None, class_name=None, overwrite=False
):
    """
    Register technique classes from a Python file and copy to custom techniques directory.

    Args:
        registry: TechniqueRegistry instance
        file_path: Path to the Python file
        custom_name: Custom name to register the technique under
        class_name: Name of specific class to register
        overwrite: Whether to overwrite existing techniques

    Returns:
        list: Names of successfully registered techniques
    """
    registered_techniques = []

    try:
        # Create custom techniques directory if it doesn't exist
        custom_dir = Path.home() / ".balancr" / "custom_techniques"
        custom_dir.mkdir(parents=True, exist_ok=True)

        # Import the module dynamically
        module_name = file_path.stem
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            logging.error(f"Could not load module from {file_path}")
            return registered_techniques

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find all classes that inherit from BaseBalancer
        technique_classes = []
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                obj.__module__
                == module_name  # Only consider classes defined in this module
                and issubclass(obj, BaseBalancer)
                and obj is not BaseBalancer
            ):  # Exclude the base class itself
                technique_classes.append((name, obj))

        # If no valid classes found
        if not technique_classes:
            logging.warning(f"No valid technique classes found in {file_path}")
            logging.info(
                "Classes must inherit from balancr.base.BaseBalancer"
            )
            return registered_techniques

        # If class_name is specified, filter to just that class
        if class_name:
            technique_classes = [
                (name, cls) for name, cls in technique_classes if name == class_name
            ]
            if not technique_classes:
                logging.error(
                    f"Class '{class_name}' not found in {file_path} or doesn't inherit from BaseBalancer"
                )
                return registered_techniques

        # If requesting a custom name but multiple classes found and no class_name specified
        if custom_name and len(technique_classes) > 1 and not class_name:
            logging.error(
                f"Multiple technique classes found in {file_path}, but custom name provided. "
                "Please specify which class to register with --class-name."
            )
            return registered_techniques

        # Register techniques
        for name, cls in technique_classes:
            # If this is a specifically requested class with a custom name
            if class_name and name == class_name and custom_name:
                register_name = custom_name
            else:
                register_name = name

            try:
                # Check if technique already exists
                existing_techniques = registry.list_available_techniques()
                if (
                    register_name in existing_techniques.get("custom", [])
                    and not overwrite
                ):
                    logging.warning(
                        f"Technique '{register_name}' already exists. "
                        "Use --overwrite to replace it."
                    )
                    continue

                # Register the technique
                registry.register_custom_technique(register_name, cls)
                registered_techniques.append(register_name)
                logging.info(f"Successfully registered technique: {register_name}")

            except Exception as e:
                logging.error(f"Error registering technique '{register_name}': {e}")

        # For successfully registered techniques, copy the file
        if registered_techniques:
            # Generate a unique filename (in case multiple files have same name)
            dest_file = custom_dir / f"{file_path.stem}_{hash(str(file_path))}.py"
            shutil.copy2(file_path, dest_file)
            logging.debug(f"Copied {file_path} to {dest_file}")

            # Create a metadata file to map technique names to files
            metadata_file = custom_dir / "techniques_metadata.json"
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

            # Update metadata with new techniques
            class_mapping = {cls_name: cls_name for cls_name, _ in technique_classes}
            if class_name and custom_name:
                class_mapping[custom_name] = class_name

            for technique_name in registered_techniques:
                original_class = class_mapping.get(technique_name, technique_name)
                metadata[technique_name] = {
                    "file": str(dest_file),
                    "class_name": original_class,
                    "registered_at": datetime.now().isoformat(),
                }

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")

    return registered_techniques


def _remove_techniques(args):
    """
    Remove custom techniques as specified in the args.

    Args:
        args: Command line arguments from argparse

    Returns:
        int: Exit code
    """
    # Get path to custom techniques directory
    custom_dir = Path.home() / ".balancr" / "custom_techniques"
    metadata_file = custom_dir / "techniques_metadata.json"

    # Check if metadata file exists
    if not metadata_file.exists():
        logging.error("No custom techniques have been registered.")
        return 1

    # Load metadata
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # If no custom techniques
    if not metadata:
        logging.error("No custom techniques have been registered.")
        return 1

    # Remove all custom techniques
    if args.remove_all:
        logging.info("Removing all custom techniques...")

        # Remove all technique files
        file_paths = set(info["file"] for info in metadata.values())
        for file_path in file_paths:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception as e:
                logging.warning(f"Error removing file {file_path}: {e}")

        # Clear metadata
        metadata = {}
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print("All custom techniques have been removed.")
        return 0

    # Remove specific techniques
    removed_techniques = []
    for technique_name in args.remove:
        if technique_name in metadata:
            # Note the file path (we'll check if it's used by other techniques)
            file_path = metadata[technique_name]["file"]

            # Remove from metadata
            del metadata[technique_name]
            removed_techniques.append(technique_name)

            # Check if the file is still used by other techniques
            file_still_used = any(
                info["file"] == file_path for info in metadata.values()
            )

            # If not used, remove the file
            if not file_still_used:
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception as e:
                    logging.warning(f"Error removing file {file_path}: {e}")
        else:
            logging.warning(f"Technique '{technique_name}' not found.")

    # Save updated metadata
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    if removed_techniques:
        print("\nRemoved techniques:")
        for technique in removed_techniques:
            print(f"  - {technique}")
        return 0
    else:
        logging.error("No matching techniques were found.")
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

    # Print custom classifiers if any
    if "custom" in classifiers and classifiers["custom"]:
        print("\nCustom Classifiers:")
        for module_name, clf_list in classifiers["custom"].items():
            if clf_list:
                print(f"\n  {module_name.capitalize()}:")
                for clf in sorted(clf_list):
                    print(f"    - {clf}")

    # Print sklearn classifiers by module
    if "sklearn" in classifiers:
        print("\nScikit-learn Classifiers:")
        for module_name, clf_list in classifiers["sklearn"].items():
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


def register_classifiers(args):
    """
    Handle the register-classifiers command.

    This command allows users to register custom classifiers from
    Python files or folders, or remove existing custom classifiers.

    Args:
        args: Command line arguments from argparse

    Returns:
        int: Exit code
    """
    try:
        # Ensure framework is available
        if ClassifierRegistry is None:
            logging.error(
                "Classifier registry not available. Please check installation."
            )
            return 1

        # Handle removal operations
        if args.remove or args.remove_all:
            return _remove_classifiers(args)

        registry = ClassifierRegistry()

        # Track successfully registered classifiers
        registered_classifiers = []

        # Process file path
        if args.file_path:
            file_path = Path(args.file_path)

            if not file_path.exists():
                logging.error(f"File not found: {file_path}")
                return 1

            if not file_path.is_file() or file_path.suffix.lower() != ".py":
                logging.error(f"Not a Python file: {file_path}")
                return 1

            # Register classifiers from the file
            registered = _register_classifier_from_file(
                registry, file_path, args.name, args.class_name, args.overwrite
            )
            registered_classifiers.extend(registered)

        # Process folder path
        elif args.folder_path:
            folder_path = Path(args.folder_path)

            if not folder_path.exists():
                logging.error(f"Folder not found: {folder_path}")
                return 1

            if not folder_path.is_dir():
                logging.error(f"Not a directory: {folder_path}")
                return 1

            # Register classifiers from all Python files in the folder
            for py_file in folder_path.glob("*.py"):
                registered = _register_classifier_from_file(
                    registry,
                    py_file,
                    None,  # Don't use custom name for folder scanning
                    None,  # Don't use class name for folder scanning
                    args.overwrite,
                )
                registered_classifiers.extend(registered)

        # Print summary
        if registered_classifiers:
            print("\nSuccessfully registered classifiers:")
            for classifier in registered_classifiers:
                print(f"  - {classifier}")

            print("\nYou can now use these classifiers in comparisons. For example:")
            print(f"  balancr select-classifiers {registered_classifiers[0]}")
            return 0
        else:
            logging.warning("No valid classifiers found to register.")
            return 1

    except Exception as e:
        logging.error(f"Error registering classifiers: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def _register_classifier_from_file(
    registry, file_path, custom_name=None, class_name=None, overwrite=False
):
    """
    Register classifier classes from a Python file.

    Args:
        registry: ClassifierRegistry instance
        file_path: Path to the Python file
        custom_name: Custom name to register the classifier under
        class_name: Name of specific class to register
        overwrite: Whether to overwrite existing classifiers

    Returns:
        list: Names of successfully registered classifiers
    """
    registered_classifiers = []

    try:
        # Create custom classifiers directory if it doesn't exist
        custom_dir = Path.home() / ".balancr" / "custom_classifiers"
        custom_dir.mkdir(parents=True, exist_ok=True)

        # Import the module dynamically
        module_name = file_path.stem
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            logging.error(f"Could not load module from {file_path}")
            return registered_classifiers

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Ensure sklearn BaseEstimator is available
        try:
            from sklearn.base import BaseEstimator
        except ImportError:
            logging.error(
                "scikit-learn is not available. Please install it using: pip install scikit-learn"
            )
            return registered_classifiers

        # Find all classes that inherit from BaseEstimator and have fit/predict methods
        classifier_classes = []
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                obj.__module__
                == module_name  # Only consider classes defined in this module
                and issubclass(obj, BaseEstimator)
                and hasattr(obj, "fit")
                and hasattr(obj, "predict")
            ):
                classifier_classes.append((name, obj))

        # If no valid classes found
        if not classifier_classes:
            logging.warning(f"No valid classifier classes found in {file_path}")
            logging.info(
                "Classes must inherit from sklearn.base.BaseEstimator and implement fit and predict methods"
            )
            return registered_classifiers

        # If class_name is specified, filter to just that class
        if class_name:
            classifier_classes = [
                (name, cls) for name, cls in classifier_classes if name == class_name
            ]
            if not classifier_classes:
                logging.error(
                    f"Class '{class_name}' not found in {file_path} or doesn't meet classifier requirements"
                )
                return registered_classifiers

        # If requesting a custom name but multiple classes found and no class_name specified
        if custom_name and len(classifier_classes) > 1 and not class_name:
            logging.error(
                f"Multiple classifier classes found in {file_path}, but custom name provided. "
                "Please specify which class to register with --class-name."
            )
            return registered_classifiers

        # Register the classifiers
        for name, cls in classifier_classes:
            # If this is a specifically requested class with a custom name
            if class_name and name == class_name and custom_name:
                register_name = custom_name
            else:
                register_name = name

            try:
                # Check if classifier already exists
                existing_classifiers = registry.list_available_classifiers()
                flat_existing = []
                for module_classifiers in existing_classifiers.get(
                    "custom", {}
                ).values():
                    flat_existing.extend(module_classifiers)

                if register_name in flat_existing and not overwrite:
                    logging.warning(
                        f"Classifier '{register_name}' already exists. "
                        "Use --overwrite to replace it."
                    )
                    continue

                # Register the classifier
                registry.register_custom_classifier(register_name, cls)
                registered_classifiers.append(register_name)
                logging.info(f"Successfully registered classifier: {register_name}")

            except Exception as e:
                logging.error(f"Error registering classifier '{register_name}': {e}")

        # For successfully registered classifiers, copy the file
        if registered_classifiers:
            # Generate a unique filename (in case multiple files have same name)
            dest_file = custom_dir / f"{file_path.stem}_{hash(str(file_path))}.py"
            shutil.copy2(file_path, dest_file)
            logging.debug(f"Copied {file_path} to {dest_file}")

            # Create a metadata file to map classifier names to files
            metadata_file = custom_dir / "classifiers_metadata.json"
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

            # Update metadata with new classifiers
            class_mapping = {cls_name: cls_name for cls_name, _ in classifier_classes}
            if class_name and custom_name:
                class_mapping[custom_name] = class_name

            for classifier_name in registered_classifiers:
                original_class = class_mapping.get(classifier_name, classifier_name)
                metadata[classifier_name] = {
                    "file": str(dest_file),
                    "class_name": original_class,
                    "registered_at": datetime.now().isoformat(),
                }

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")

    return registered_classifiers


def _remove_classifiers(args):
    """
    Remove custom classifiers as specified in the args.

    Args:
        args: Command line arguments from argparse

    Returns:
        int: Exit code
    """
    # Get path to custom classifiers directory
    custom_dir = Path.home() / ".balancr" / "custom_classifiers"
    metadata_file = custom_dir / "classifiers_metadata.json"

    # Check if metadata file exists
    if not metadata_file.exists():
        logging.error("No custom classifiers have been registered.")
        return 1

    # Load metadata
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # If no custom classifiers
    if not metadata:
        logging.error("No custom classifiers have been registered.")
        return 1

    # Remove all custom classifiers
    if args.remove_all:
        logging.info("Removing all custom classifiers...")

        # Remove all classifier files
        file_paths = set(info["file"] for info in metadata.values())
        for file_path in file_paths:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception as e:
                logging.warning(f"Error removing file {file_path}: {e}")

        # Clear metadata
        metadata = {}
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print("All custom classifiers have been removed.")
        return 0

    # Remove specific classifiers
    removed_classifiers = []
    for classifier_name in args.remove:
        if classifier_name in metadata:
            # Note the file path (we'll check if it's used by other classifiers)
            file_path = metadata[classifier_name]["file"]

            # Remove from metadata
            del metadata[classifier_name]
            removed_classifiers.append(classifier_name)

            # Check if the file is still used by other classifiers
            file_still_used = any(
                info["file"] == file_path for info in metadata.values()
            )

            # If not used, remove the file
            if not file_still_used:
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception as e:
                    logging.warning(f"Error removing file {file_path}: {e}")
        else:
            logging.warning(f"Classifier '{classifier_name}' not found.")

    # Save updated metadata
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    if removed_classifiers:
        print("\nRemoved classifiers:")
        for classifier in removed_classifiers:
            print(f"  - {classifier}")
        return 0
    else:
        logging.error("No matching classifiers were found.")
        return 1


def configure_metrics(args):
    """
    Handle the configure-metrics command.

    Args:
        args: Command line arguments from argparse

    Returns:
        int: Exit code
    """
    # Define all available metrics
    all_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "specificity",
        "g_mean",
        "average_precision",
    ]

    # If 'all' is specified, use all available metrics
    if "all" in args.metrics:
        metrics_to_use = all_metrics
        metrics_str = "all available metrics"
    else:
        metrics_to_use = args.metrics
        metrics_str = ", ".join(args.metrics)

    logging.info(f"Configuring metrics: {metrics_str}")

    # Update configuration with metrics settings
    settings = {
        "output": {"metrics": metrics_to_use, "save_metrics_formats": args.save_formats}
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
        print(f"  Metrics: {metrics_str}")
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
            "learning_curve_points": args.learning_curve_points,
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


def format_time(seconds):
    """Format time in seconds to minutes and seconds"""
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}mins, {remaining_seconds:.2f}secs"


def run_comparison(args):
    """
    Handle the run command.

    Args:
        args: Command line arguments from argparse

    Returns:
        int: Exit code
    """
    start_time_total = time.time()

    # Load current configuration
    try:
        current_config = config.load_config(args.config_path)
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        return 1

    # Check if all required settings are configured
    required_settings = ["data_file", "target_column", "balancing_techniques"]
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
    include_original = current_config.get("include_original_data", False)

    balancing_techniques = current_config.get("balancing_techniques", {})
    technique_names = list(balancing_techniques.keys())
    logging.info(f"Running comparison with techniques: {', '.join(technique_names)}")
    logging.info(f"Results will be saved to: {output_dir}")

    try:
        # Initialise the framework
        framework = BalancingFramework()

        # Load data
        start_time = time.time()
        logging.info(f"Loading data from {current_config['data_file']}")
        feature_columns = current_config.get("feature_columns", None)
        framework.load_data(
            current_config["data_file"],
            current_config["target_column"],
            feature_columns,
        )
        load_time = time.time() - start_time
        logging.info(f"Data loading completed (Time Taken: {format_time(load_time)})")

        # Apply preprocessing if configured
        if "preprocessing" in current_config:
            logging.info("Applying preprocessing...")
            preproc = current_config["preprocessing"]

            handle_missing = preproc.get("handle_missing", "mean")
            scale = preproc.get("scale", "standard")
            categorical_features = preproc.get("categorical_features", {})
            handle_constant_features = preproc.get("handle_constant_features", None)
            handle_correlations = preproc.get("handle_correlations", None)
            save_preprocessed_file = preproc.get("save_preprocessed", True)

            # Extract hash components information
            hash_components_dict = {}
            for feature, encoding in categorical_features.items():
                if isinstance(encoding, list) and encoding[0] == "hash":
                    hash_components_dict[feature] = encoding[
                        1
                    ]  # Store the n_components value

            framework.preprocess_data(
                handle_missing=handle_missing,
                scale=scale,
                categorical_features=categorical_features,
                hash_components_dict=hash_components_dict,
                handle_constant_features=handle_constant_features,
                handle_correlations=handle_correlations,
            )
            logging.info("Data preprocessing applied")

            # Save preprocessed dataset to a new file
            if save_preprocessed_file:
                try:
                    if "data_file" in current_config and current_config["data_file"]:
                        original_path = Path(current_config["data_file"])
                        preprocessed_path = (
                            original_path.parent
                            / f"{original_path.stem}_preprocessed{original_path.suffix}"
                        )

                        # Copy DataFrame with the preprocessed data
                        preprocessed_df = framework.X.copy()

                        # Add the target column
                        target_column = current_config.get("target_column")
                        if target_column:
                            preprocessed_df[target_column] = framework.y

                        # Save to the new file
                        if original_path.suffix.lower() == ".csv":
                            preprocessed_df.to_csv(preprocessed_path, index=False)
                        elif original_path.suffix.lower() in [".xlsx", ".xls"]:
                            preprocessed_df.to_excel(preprocessed_path, index=False)

                        logging.info(
                            f"Saved preprocessed dataset to: {preprocessed_path}"
                        )
                except Exception as e:
                    logging.warning(f"Could not save preprocessed dataset: {e}")

        # Apply balancing techniques
        start_time = time.time()
        logging.info("Applying balancing techniques...")
        framework.apply_balancing_techniques(
            technique_names,
            test_size=test_size,
            random_state=random_state,
            technique_params=balancing_techniques,
            include_original=include_original,
        )
        balancing_time = time.time() - start_time
        logging.info(
            f"Balancing techniques applied successfully (Time Taken: {format_time(balancing_time)})"
        )

        # Save balanced datasets at the root level
        balanced_dir = output_dir / "balanced_datasets"
        balanced_dir.mkdir(exist_ok=True)
        logging.info(f"Saving balanced datasets to {balanced_dir}")
        framework.generate_balanced_data(
            folder_path=str(balanced_dir),
            techniques=technique_names,
            file_format="csv",
        )

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
                    f"Generating imbalanced class distribution in {format_type} format..."
                )
                imbalanced_plot_path = (
                    output_dir / f"imbalanced_class_distribution.{format_type}"
                )
                framework.inspect_class_distribution(
                    save_path=str(imbalanced_plot_path), display=display_visualisations
                )
                logging.info(
                    f"Imbalanced class distribution saved to {imbalanced_plot_path}"
                )

                # Balanced class distributions comparison
                logging.info(
                    f"Generating balanced class distribution comparison in {format_type} format..."
                )
                balanced_plot_path = (
                    output_dir / f"balanced_class_distribution.{format_type}"
                )
                framework.compare_balanced_class_distributions(
                    save_path=str(balanced_plot_path),
                    display=display_visualisations,
                )
                logging.info(
                    f"Balanaced class distribution comparison saved to {balanced_plot_path}"
                )

        # Train classifiers
        start_time = time.time()
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

        training_time = time.time() - start_time
        logging.info(
            f"Training classifiers complete (Time Taken: {format_time(training_time)})"
        )

        # Process each classifier and save its results in a separate directory
        orig_start_time = time.time()
        for classifier_name in current_config.get("classifiers", {}):
            logging.info(f"Processing results for classifier: {classifier_name}")

            # Create classifier-specific directory
            classifier_dir = output_dir / classifier_name
            classifier_dir.mkdir(exist_ok=True)

            # Create original dataset metrics directory
            orig_metrics_dir = classifier_dir / "metrics_on_original_test"
            orig_metrics_dir.mkdir(exist_ok=True)

            # Save original dataset metrics in requested formats
            for format_type in save_metrics_formats:
                if format_type == "none":
                    continue

                results_file = orig_metrics_dir / f"comparison_results.{format_type}"
                logging.info(
                    f"Saving metrics from testing against original test data for {classifier_name} to {results_file}"
                )

                # We need a modified save_results method that can extract a specific classifier's results
                framework.save_classifier_results(
                    results_file,
                    classifier_name=classifier_name,
                    metric_type="standard_metrics",
                    file_type=format_type,
                )

            # Generate and save original test data metrics visualisations
            for format_type in save_vis_formats:
                if format_type == "none":
                    continue

                if "metrics" in vis_types_to_generate or "all" in visualisations:
                    metrics_path = orig_metrics_dir / f"metrics_comparison.{format_type}"
                    logging.info(
                        f"Generating metrics comparison for {classifier_name}, against original test data,"
                        f"in {format_type} format..."
                    )

                    metrics_to_plot = current_config.get("output", {}).get(
                        "metrics", ["precision", "recall", "f1", "roc_auc"]
                    )
                    # Call a modified plot_comparison_results that can handle specific classifier data
                    plot_comparison_results(
                        results,
                        classifier_name=classifier_name,
                        metric_type="standard_metrics",
                        metrics_to_plot=metrics_to_plot,
                        save_path=str(metrics_path),
                        display=display_visualisations,
                    )

                    if "radar" in vis_types_to_generate or "all" in visualisations:
                        std_radar_path = (
                            classifier_dir / f"metrics_on_original_test_radar.{format_type}"
                        )
                        plot_radar_chart(
                            results,
                            classifier_name=classifier_name,
                            metric_type="standard_metrics",
                            metrics_to_plot=metrics_to_plot,
                            save_path=std_radar_path,
                            display=display_visualisations,
                        )

                    if "3d" in vis_types_to_generate or "all" in visualisations:
                        std_3d_path = output_dir / "metrics_on_original_test_3d.html"
                        plot_3d_scatter(
                            results,
                            metric_type="standard_metrics",
                            metrics_to_plot=metrics_to_plot,
                            save_path=std_3d_path,
                            display=display_visualisations,
                        )

                if (
                    "learning_curves" in vis_types_to_generate
                    or "all" in visualisations
                ):
                    orig_learning_curve_path = (
                        orig_metrics_dir / f"learning_curves.{format_type}"
                    )

                    start_time = time.time()
                    logging.info(
                        f"Generating learning curves for {classifier_name}, against original test data "
                        f"in {format_type} format..."
                    )

                    # Get learning curve parameters from config
                    learning_curve_points = eval_config.get(
                        "learning_curve_points", 10
                    )
                    learning_curve_folds = eval_config.get(
                        "learning_curve_folds", 5
                    )
                    train_sizes = np.linspace(0.1, 1.0, learning_curve_points)

                    learning_curve_type = "Original Dataset"
                    framework.generate_learning_curves(
                        classifier_name=classifier_name,
                        learning_curve_type=learning_curve_type,
                        train_sizes=train_sizes,
                        n_folds=learning_curve_folds,
                        save_path=str(orig_learning_curve_path),
                        display=display_visualisations,
                    )
                    cv_learning_curves_time = time.time() - start_time
                    logging.info(
                        f"Successfully generated cv learning curves for {classifier_name}, against original test data "
                        f"(Time Taken: {format_time(cv_learning_curves_time)})"
                    )

        orig_total_time = time.time() - orig_start_time
        logging.info(
            f"Metrics evaluation against original test data total time: {format_time(orig_total_time)}"
        )

        # If cross-validation is enabled, create CV metrics directory and save results
        if cv_enabled:
            cv_start_time = time.time()
            for classifier_name in current_config.get("classifiers", {}):
                # Create classifier-specific directory
                classifier_dir = output_dir / classifier_name
                classifier_dir.mkdir(exist_ok=True)

                cv_metrics_dir = classifier_dir / "metrics_on_balanced_cv"
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
                            f"Generating CV metrics comparison for {classifier_name} in {format_type} format..."
                        )

                        metrics_to_plot = current_config.get("output", {}).get(
                            "metrics", ["precision", "recall", "f1", "roc_auc"]
                        )
                        plot_comparison_results(
                            results,
                            classifier_name=classifier_name,
                            metric_type="cv_metrics",
                            metrics_to_plot=metrics_to_plot,
                            save_path=str(cv_metrics_path),
                            display=display_visualisations,
                        )

                        if "radar" in vis_types_to_generate or "all" in visualisations:
                            cv_radar_path = (
                                classifier_dir / f"cv_metrics_radar.{format_type}"
                            )
                            plot_radar_chart(
                                results,
                                classifier_name=classifier_name,
                                metric_type="cv_metrics",
                                metrics_to_plot=metrics_to_plot,
                                save_path=cv_radar_path,
                                display=display_visualisations,
                            )

                        if "3d" in vis_types_to_generate or "all" in visualisations:
                            cv_3d_path = output_dir / "cv_metrics_3d.html"
                            plot_3d_scatter(
                                results,
                                metric_type="cv_metrics",
                                metrics_to_plot=metrics_to_plot,
                                save_path=cv_3d_path,
                                display=display_visualisations,
                            )

                    if (
                        "learning_curves" in vis_types_to_generate
                        or "all" in visualisations
                    ):
                        cv_learning_curve_path = (
                            cv_metrics_dir / f"learning_curves.{format_type}"
                        )

                        start_time = time.time()
                        logging.info(
                            f"Generating CV learning curves for {classifier_name} in {format_type} format..."
                        )

                        # Get learning curve parameters from config
                        learning_curve_points = eval_config.get(
                            "learning_curve_points", 10
                        )
                        learning_curve_folds = eval_config.get(
                            "learning_curve_folds", 5
                        )
                        train_sizes = np.linspace(0.1, 1.0, learning_curve_points)

                        learning_curve_type = "Balanced Datasets"
                        framework.generate_learning_curves(
                            classifier_name=classifier_name,
                            learning_curve_type=learning_curve_type,
                            train_sizes=train_sizes,
                            n_folds=learning_curve_folds,
                            save_path=str(cv_learning_curve_path),
                            display=display_visualisations,
                        )
                        cv_learning_curves_time = time.time() - start_time
                        logging.info(
                            f"Successfully generated cv learning curves for {classifier_name}"
                            f"(Time Taken: {format_time(cv_learning_curves_time)})"
                        )
            cv_total_time = time.time() - cv_start_time
            logging.info(
                f"Cross validation metrics evaluation total time: {format_time(cv_total_time)}"
            )

        total_time = time.time() - start_time_total
        logging.info(f"Total execution time: {format_time(total_time)}")

        # Print summary of timing results
        print("\nExecution Time Summary:\n")
        print(f"  Data Loading:                                               {format_time(load_time)}")
        print(f"  Balancing:                                                  {format_time(balancing_time)}")
        print(f"  Training Classifiers:                                       {format_time(training_time)}")
        print(f"  Metrics From Testing Against Original Test Dataset:         {format_time(orig_total_time)}")
        if cv_enabled:
            print(f"  CV Metrics From Testing Against Balanced Test Datasets:     {format_time(cv_total_time)}")
        print(f"  Total Time:                                                 {format_time(total_time)}")

        print("\nResults Summary:")

        # Check and print Standard Metrics if available
        has_standard_metrics = any(
            "standard_metrics" in technique_metrics
            and any(m in metrics for m in technique_metrics["standard_metrics"])
            for classifier_results in results.values()
            for technique_metrics in classifier_results.values()
        )

        if has_standard_metrics:
            print("\nMetrics From Testing Against Original Test Dataset:")
            for classifier_name, classifier_results in results.items():
                print(f"\n{classifier_name}:")
                for technique_name, technique_metrics in classifier_results.items():
                    if "standard_metrics" in technique_metrics:
                        std_metrics = technique_metrics["standard_metrics"]
                        if any(m in metrics for m in std_metrics):
                            print(f"  {technique_name}:")
                            for metric_name, value in std_metrics.items():
                                if metric_name in metrics:
                                    print(f"    {metric_name}: {value:.4f}")

        # Check and print Cross Validation Metrics if available
        has_cv_metrics = any(
            "cv_metrics" in technique_metrics
            and any(
                metric_name.startswith("cv_")
                and metric_name[len("cv_"):].rsplit("_", 1)[0] in metrics
                for metric_name in technique_metrics["cv_metrics"]
            )
            for classifier_results in results.values()
            for technique_metrics in classifier_results.values()
        )

        if has_cv_metrics:
            print("\nCross Validation Metrics From Testing Against Balanced Test Datasets:")
            for classifier_name, classifier_results in results.items():
                print(f"\n{classifier_name}:")
                for technique_name, technique_metrics in classifier_results.items():
                    if "cv_metrics" in technique_metrics:
                        cv_metrics = technique_metrics["cv_metrics"]

                        # Check if any relevant cv metric exists for this technique
                        if any(
                            metric_name.startswith("cv_")
                            and metric_name[len("cv_"):].rsplit("_", 1)[0] in metrics
                            for metric_name in cv_metrics
                        ):
                            print(f"  {technique_name}:")

                            # Now print only relevant metrics
                            for metric_name, value in cv_metrics.items():
                                if metric_name.startswith("cv_"):
                                    base_name = metric_name[len("cv_"):].rsplit(
                                        "_", 1
                                    )[0]
                                    if base_name in metrics:
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
