from pathlib import Path
from typing import Dict, Type, Optional, List
import importlib
import inspect
import logging
import json
from sklearn.base import BaseEstimator


class ClassifierRegistry:
    """Registry for managing classification algorithms from various sources"""

    # List of scikit-learn modules where we'll look for classifiers
    SKLEARN_MODULES = [
        "sklearn.ensemble",
        "sklearn.linear_model",
        "sklearn.tree",
        "sklearn.svm",
        "sklearn.neighbors",
        "sklearn.naive_bayes",
        "sklearn.neural_network",
        "sklearn.discriminant_analysis",
    ]

    def __init__(self):
        # Storage for custom classifiers
        self.custom_classifiers: Dict[str, Type[BaseEstimator]] = {}

        # Cache of sklearn classifiers, organised by module
        self._cached_sklearn_classifiers: Dict[str, Dict[str, tuple]] = {}

        # Find all available classifiers when initialised
        self._discover_sklearn_classifiers()

        self._load_custom_classifiers()

    def _discover_sklearn_classifiers(self) -> None:
        """Look through scikit-learn modules to find usable classifier classes"""
        for module_path in self.SKLEARN_MODULES:
            try:
                # Try to import module
                module = importlib.import_module(module_path)

                # Get just the module name (e.g., 'ensemble' from 'sklearn.ensemble')
                module_name = module_path.split(".")[-1]

                # Make sure we have a dict ready for this module
                if module_name not in self._cached_sklearn_classifiers:
                    self._cached_sklearn_classifiers[module_name] = {}

                # Look at all classes in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # We consider something a classifier if it:
                    # 1. Has fit and predict methods
                    # 2. Inherits from BaseEstimator
                    if (
                        hasattr(obj, "fit")
                        and hasattr(obj, "predict")
                        and issubclass(obj, BaseEstimator)
                    ):

                        # Skip abstract base classes and internal classes
                        if not name.startswith("Base") and not name.startswith("_"):
                            self._cached_sklearn_classifiers[module_name][name] = (
                                module_path,
                                obj,
                            )

            except ImportError as e:
                logging.warning(f"Couldn't import {module_path}: {str(e)}")

    def get_classifier_class(
        self, classifier_name: str, module_name: Optional[str] = None
    ) -> Optional[Type[BaseEstimator]]:
        """
        Find a classifier class by its name, handling suffixed variations.

        Args:
            classifier_name: Name of the classifier (e.g., 'RandomForestClassifier')
            module_name: Optional module to look in (e.g., 'ensemble', 'linear_model')

        Returns:
            The classifier class if found, None otherwise
        """
        # First, check for exact matches in custom classifiers
        if classifier_name in self.custom_classifiers:
            return self.custom_classifiers[classifier_name]

        # If user specified a module, only look there for exact match first
        if module_name is not None:
            if (
                module_name in self._cached_sklearn_classifiers
                and classifier_name in self._cached_sklearn_classifiers[module_name]
            ):
                _, classifier_class = self._cached_sklearn_classifiers[module_name][
                    classifier_name
                ]
                return classifier_class

        # Otherwise, look through all modules for exact match
        if module_name is None:
            for module_dict in self._cached_sklearn_classifiers.values():
                if classifier_name in module_dict:
                    _, classifier_class = module_dict[classifier_name]
                    return classifier_class

        # If no exact match, extract base name if this is a variation with _ or - suffix
        base_name = None
        for delimiter in ["_", "-"]:
            if delimiter in classifier_name:
                parts = classifier_name.split(delimiter, 1)
                if len(parts) > 1 and parts[0]:  # Ensure we have a non-empty base name
                    base_name = parts[0]
                    break

        # If we have a valid base name, look it up
        if base_name:
            # Check custom classifiers for the base name
            if base_name in self.custom_classifiers:
                return self.custom_classifiers[base_name]

            # If user specified a module, only look there for the base name
            if module_name is not None:
                if (
                    module_name in self._cached_sklearn_classifiers
                    and base_name in self._cached_sklearn_classifiers[module_name]
                ):
                    _, classifier_class = self._cached_sklearn_classifiers[module_name][
                        base_name
                    ]
                    return classifier_class
            else:
                # Otherwise, look through all modules for the base name
                for module_dict in self._cached_sklearn_classifiers.values():
                    if base_name in module_dict:
                        _, classifier_class = module_dict[base_name]
                        return classifier_class

        # If not found, try to discover new techniques (in case sklearn was updated)
        self._discover_sklearn_classifiers()

        # Try exact match again with freshly discovered classifiers
        if module_name is not None:
            if (
                module_name in self._cached_sklearn_classifiers
                and classifier_name in self._cached_sklearn_classifiers[module_name]
            ):
                _, classifier_class = self._cached_sklearn_classifiers[module_name][
                    classifier_name
                ]
                return classifier_class
        else:
            for module_dict in self._cached_sklearn_classifiers.values():
                if classifier_name in module_dict:
                    _, classifier_class = module_dict[classifier_name]
                    return classifier_class

        # Try base name again with freshly discovered classifiers
        if base_name:
            if module_name is not None:
                if (
                    module_name in self._cached_sklearn_classifiers
                    and base_name in self._cached_sklearn_classifiers[module_name]
                ):
                    _, classifier_class = self._cached_sklearn_classifiers[module_name][
                        base_name
                    ]
                    return classifier_class
            else:
                for module_dict in self._cached_sklearn_classifiers.values():
                    if base_name in module_dict:
                        _, classifier_class = module_dict[base_name]
                        return classifier_class

        return None

    def list_available_classifiers(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Get a hierarchical list of all available classifiers.

        Returns:
            Dictionary organised by source -> module -> classifier names
        """
        # Refresh cache in case new classifiers were installed
        self._discover_sklearn_classifiers()

        result = {"custom": {}, "sklearn": self._get_sklearn_classifiers_by_module()}

        # Add custom classifiers if there are any
        if self.custom_classifiers:
            result["custom"] = {"general": list(self.custom_classifiers.keys())}

        return result

    def _get_sklearn_classifiers_by_module(self) -> Dict[str, List[str]]:
        """Organise scikit-learn classifiers by their module for a cleaner display"""
        result = {}

        for module_name, classifiers in self._cached_sklearn_classifiers.items():
            if classifiers:  # Only include modules that have classifiers
                result[module_name] = list(classifiers.keys())

        return result

    def register_custom_classifier(
        self, name: str, classifier_class: Type[BaseEstimator]
    ) -> None:
        """
        Register a custom classifier for use in the framework.

        Args:
            name: Name to register the classifier under
            classifier_class: The classifier class itself

        Raises:
            TypeError: If the classifier doesn't meet requirements
            ValueError: If the name is invalid
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Classifier name must be a non-empty string")

        if classifier_class is None:
            raise TypeError("Classifier class cannot be None")

        if not isinstance(classifier_class, type) or not issubclass(
            classifier_class, BaseEstimator
        ):
            raise TypeError(
                "Classifier class must inherit from sklearn.base.BaseEstimator"
            )

        # Make sure it has the required methods
        if not hasattr(classifier_class, "fit") or not hasattr(
            classifier_class, "predict"
        ):
            raise TypeError(
                "Classifier class must implement 'fit' and 'predict' methods"
            )

        self.custom_classifiers[name] = classifier_class

    def _load_custom_classifiers(self) -> None:
        """Load registered custom classifiers from the custom classifiers directory."""
        custom_dir = Path.home() / ".balancr" / "custom_classifiers"
        if not custom_dir.exists():
            return

        metadata_file = custom_dir / "classifiers_metadata.json"
        if not metadata_file.exists():
            return

        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            for classifier_name, info in metadata.items():
                file_path = Path(info["file"])
                class_name = info["class_name"]

                if not file_path.exists():
                    logging.warning(f"Custom classifier file not found: {file_path}")
                    continue

                try:
                    # Import the module dynamically
                    module_name = file_path.stem
                    spec = importlib.util.spec_from_file_location(
                        module_name, file_path
                    )
                    if spec is None or spec.loader is None:
                        logging.warning(f"Could not load module from {file_path}")
                        continue

                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find the specific class
                    classifier_class = None
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (
                            name == class_name
                            and hasattr(obj, "fit")
                            and hasattr(obj, "predict")
                        ):
                            classifier_class = obj
                            break

                    if classifier_class:
                        self.custom_classifiers[classifier_name] = classifier_class
                        logging.debug(f"Loaded custom classifier: {classifier_name}")
                    else:
                        logging.warning(f"Class {class_name} not found in {file_path}")

                except Exception as e:
                    logging.warning(
                        f"Error loading custom classifier {classifier_name}: {e}"
                    )

        except Exception as e:
            logging.warning(f"Error loading custom classifiers metadata: {e}")
