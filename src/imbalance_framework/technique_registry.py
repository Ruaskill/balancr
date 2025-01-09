from typing import Dict, Type, Optional
import importlib
import inspect
from .base import BaseBalancer


class TechniqueRegistry:
    """Registry for managing balancing techniques from various sources"""

    # Define the modules to check for techniques
    IMBLEARN_MODULES = [
        "imblearn.over_sampling",
        "imblearn.under_sampling",
        "imblearn.combine",
    ]

    def __init__(self):
        self.custom_techniques: Dict[str, Type[BaseBalancer]] = {}
        self._cached_imblearn_techniques: Dict[str, tuple] = {}
        self._discover_imblearn_techniques()

    def _discover_imblearn_techniques(self) -> None:
        """Dynamically discover all available techniques in imblearn"""
        for module_path in self.IMBLEARN_MODULES:
            try:
                module = importlib.import_module(module_path)
                # Get all classes from the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Check if it's a sampler (has fit_resample method)
                    if hasattr(obj, "fit_resample"):
                        self._cached_imblearn_techniques[name] = (module_path, obj)
            except ImportError as e:
                print(f"Warning: Could not import {module_path}: {str(e)}")

    def get_technique_class(self, technique_name: str) -> Optional[Type[BaseBalancer]]:
        """Get the technique class by name"""
        # Check custom techniques first
        if technique_name in self.custom_techniques:
            return self.custom_techniques[technique_name]

        # Check imblearn techniques
        if technique_name in self._cached_imblearn_techniques:
            module_path, technique_class = self._cached_imblearn_techniques[
                technique_name
            ]
            return self._wrap_imblearn_technique(technique_class)

        # If not found, try to discover new techniques (in case imblearn was updated)
        self._discover_imblearn_techniques()
        if technique_name in self._cached_imblearn_techniques:
            module_path, technique_class = self._cached_imblearn_techniques[
                technique_name
            ]
            return self._wrap_imblearn_technique(technique_class)

        return None

    def list_available_techniques(self) -> Dict[str, list]:
        """List all available techniques grouped by source"""
        # Rediscover techniques in case new ones were added
        self._discover_imblearn_techniques()

        return {
            "custom": list(self.custom_techniques.keys()),
            "imblearn": list(self._cached_imblearn_techniques.keys()),
        }

    def register_custom_technique(
        self, name: str, technique_class: Type[BaseBalancer]
    ) -> None:
        """
        Register a custom balancing technique.

        Args:
            name: Name of the technique
            technique_class: Class implementing the balancing technique

        Raises:
            TypeError: If technique_class is None or doesn't inherit from BaseBalancer
            ValueError: If name is empty or not a string
        """
        # Error handling
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Technique name must be a non-empty string")

        if technique_class is None:
            raise TypeError("Technique class cannot be None")

        if not isinstance(technique_class, type) or not issubclass(
            technique_class, BaseBalancer
        ):
            raise TypeError("Technique class must inherit from BaseBalancer")
        self.custom_techniques[name] = technique_class

    def _wrap_imblearn_technique(self, technique_class: type) -> Type[BaseBalancer]:
        """Wrap imblearn technique to conform to our BaseBalancer interface"""

        class WrappedTechnique(BaseBalancer):
            def __init__(self):
                super().__init__()
                self.technique = technique_class()

            def balance(self, X, y):
                return self.technique.fit_resample(X, y)

        return WrappedTechnique
