from typing import Dict, Type, Optional, List
import importlib
import inspect
import logging
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

    def _discover_sklearn_classifiers(self) -> None:
        """Look through scikit-learn modules to find usable classifier classes"""
        for module_path in self.SKLEARN_MODULES:
            try:
                # Try to import the module
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
        Find a classifier class by its name.

        Args:
            classifier_name: Name of the classifier (e.g., 'RandomForestClassifier')
            module_name: Optional module to look in (e.g., 'ensemble', 'linear_model')

        Returns:
            The classifier class if found, None otherwise
        """
        # Check if it's a custom classifier
        if classifier_name in self.custom_classifiers:
            return self.custom_classifiers[classifier_name]

        # If user specified a module, only look there
        if module_name is not None:
            if (
                module_name in self._cached_sklearn_classifiers
                and classifier_name in self._cached_sklearn_classifiers[module_name]
            ):
                _, classifier_class = self._cached_sklearn_classifiers[module_name][
                    classifier_name
                ]
                return classifier_class
            return None

        # Otherwise, look through all modules
        for module_dict in self._cached_sklearn_classifiers.values():
            if classifier_name in module_dict:
                _, classifier_class = module_dict[classifier_name]
                return classifier_class

        # If not found, try to discover new techniques (in case sklearn was updated)
        self._discover_sklearn_classifiers()

        # Same logic as before, but after rediscovery
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
