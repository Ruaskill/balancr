from pathlib import Path
from typing import Dict, Type, Optional, Any
import importlib
import inspect
import logging
import json
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
        self._load_custom_techniques()

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
                logging.warning(f"Could not import {module_path}: {str(e)}")

    def get_technique_class(self, technique_name: str) -> Optional[Type[BaseBalancer]]:
        """Get the technique class by name, handling suffixed variations."""
        # First, check for exact matches in custom techniques
        if technique_name in self.custom_techniques:
            return self.custom_techniques[technique_name]

        # Then check for exact matches in imblearn techniques
        if technique_name in self._cached_imblearn_techniques:
            module_path, technique_class = self._cached_imblearn_techniques[
                technique_name
            ]
            return self._wrap_imblearn_technique(technique_class)

        # If no exact match, extract base name if this is a variation with _ or - suffix
        base_name = None
        for delimiter in ["_", "-"]:
            if delimiter in technique_name:
                parts = technique_name.split(delimiter, 1)
                if len(parts) > 1 and parts[0]:  # Ensure we have a non-empty base name
                    base_name = parts[0]
                    break

        # If we have a valid base name, try to look it up
        if base_name:
            # Check custom techniques for the base name
            if base_name in self.custom_techniques:
                return self.custom_techniques[base_name]

            # Check imblearn techniques for the base name
            if base_name in self._cached_imblearn_techniques:
                module_path, technique_class = self._cached_imblearn_techniques[
                    base_name
                ]
                return self._wrap_imblearn_technique(technique_class)

        # If still not found, try to discover new techniques (in case imblearn was updated)
        self._discover_imblearn_techniques()

        # Try exact match again with freshly discovered techniques
        if technique_name in self._cached_imblearn_techniques:
            module_path, technique_class = self._cached_imblearn_techniques[
                technique_name
            ]
            return self._wrap_imblearn_technique(technique_class)

        # Try base name again with freshly discovered techniques
        if base_name and base_name in self._cached_imblearn_techniques:
            module_path, technique_class = self._cached_imblearn_techniques[base_name]
            return self._wrap_imblearn_technique(technique_class)

        return None

    def get_technique_default_params(self, technique_name: str) -> Dict[str, Any]:
        """
        Extract default parameters from a technique, handling suffixed variations.

        Args:
            technique_name: Name of the technique to extract parameters from

        Returns:
            Dictionary of parameter names and their default values
        """
        # First check for exact match in custom techniques
        if technique_name in self.custom_techniques:
            technique_class = self.custom_techniques[technique_name]
            return self._extract_params_from_class(technique_class)

        # Then check for exact match in imblearn techniques
        if technique_name in self._cached_imblearn_techniques:
            _, technique_class = self._cached_imblearn_techniques[technique_name]
            return self._extract_params_from_class(technique_class)

        # If no exact match, try to find a base name by removing suffix
        base_name = None
        for delimiter in ["_", "-"]:
            if delimiter in technique_name:
                parts = technique_name.split(delimiter, 1)
                if len(parts) > 1 and parts[0]:
                    base_name = parts[0]
                    break

        # If we have a valid base name, look it up
        if base_name:
            # Check custom techniques for the base name
            if base_name in self.custom_techniques:
                technique_class = self.custom_techniques[base_name]
                return self._extract_params_from_class(technique_class)

            # Check imblearn techniques for the base name
            if base_name in self._cached_imblearn_techniques:
                _, technique_class = self._cached_imblearn_techniques[base_name]
                return self._extract_params_from_class(technique_class)

        # If still not found
        logging.warning(
            f"Technique '{technique_name}' not found. Cannot extract parameters."
        )
        return {}

    def _extract_params_from_class(self, cls) -> Dict[str, Any]:
        """
        Extract default parameters from a class's __init__ method.

        Args:
            cls: The class to extract parameters from

        Returns:
            Dictionary of parameter names and their default values
        """
        params = {}

        try:
            # Get the signature of the __init__ method
            sig = inspect.signature(cls.__init__)

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
            logging.warning(f"Error extracting parameters from {cls.__name__}: {e}")

        return params

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

    def _load_custom_techniques(self) -> None:
        """Load registered custom techniques from the custom techniques directory."""
        custom_dir = Path.home() / ".balancr" / "custom_techniques"
        if not custom_dir.exists():
            return

        metadata_file = custom_dir / "techniques_metadata.json"
        if not metadata_file.exists():
            return

        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            for technique_name, info in metadata.items():
                file_path = Path(info["file"])
                class_name = info["class_name"]

                if not file_path.exists():
                    logging.warning(f"Custom technique file not found: {file_path}")
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
                    technique_class = None
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if name == class_name and issubclass(obj, BaseBalancer):
                            technique_class = obj
                            break

                    if technique_class:
                        self.custom_techniques[technique_name] = technique_class
                        logging.debug(f"Loaded custom technique: {technique_name}")
                    else:
                        logging.warning(f"Class {class_name} not found in {file_path}")

                except Exception as e:
                    logging.warning(
                        f"Error loading custom technique {technique_name}: {e}"
                    )

        except Exception as e:
            logging.warning(f"Error loading custom techniques metadata: {e}")

    def _wrap_imblearn_technique(self, technique_class: type) -> Type[BaseBalancer]:
        """Wrap imblearn technique to conform to our BaseBalancer interface"""

        class WrappedTechnique(BaseBalancer):
            def __init__(self, **kwargs):
                super().__init__()
                self.technique = technique_class(**kwargs)

            def balance(self, X, y):
                return self.technique.fit_resample(X, y)

        return WrappedTechnique
