import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from imbalance_framework.base import BaseBalancer
from imbalance_framework.technique_registry import TechniqueRegistry


# Mock classes
class MockSMOTE:
    def fit_resample(self, X, y):
        return X, y


class MockRandomUnderSampler:
    def fit_resample(self, X, y):
        return X, y


@pytest.fixture
def mock_imblearn_modules():
    """Mock the imblearn modules for testing"""
    mock_over = MagicMock()
    mock_over.SMOTE = MockSMOTE

    mock_under = MagicMock()
    mock_under.RandomUnderSampler = MockRandomUnderSampler

    return {
        "imblearn.over_sampling": mock_over,
        "imblearn.under_sampling": mock_under,
    }


@pytest.fixture
def registry():
    """Create a fresh registry for each test"""
    return TechniqueRegistry()


@pytest.fixture
def sample_data():
    """Create sample data for testing balancing techniques"""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    return X, y


def test_registry_initialization(registry):
    """Test that registry initializes correctly"""
    assert hasattr(registry, "custom_techniques")
    assert isinstance(registry.custom_techniques, dict)
    assert hasattr(registry, "_cached_imblearn_techniques")
    assert isinstance(registry._cached_imblearn_techniques, dict)


@patch("importlib.import_module")
def test_discover_imblearn_techniques(mock_importlib, mock_imblearn_modules):
    """Test discovery of imblearn techniques"""

    def mock_import(module_path):
        return mock_imblearn_modules.get(module_path)

    mock_importlib.side_effect = mock_import

    registry = TechniqueRegistry()
    registry._discover_imblearn_techniques()

    # Check if techniques were discovered
    assert "SMOTE" in registry._cached_imblearn_techniques
    assert "RandomUnderSampler" in registry._cached_imblearn_techniques


def test_register_custom_technique(registry):
    """Test registration of custom techniques"""

    class CustomBalancer(BaseBalancer):
        def balance(self, X, y):
            return X, y

    registry.register_custom_technique("CustomTechnique", CustomBalancer)
    assert "CustomTechnique" in registry.custom_techniques

    # Test retrieving the registered technique
    technique_class = registry.get_technique_class("CustomTechnique")
    assert technique_class is not None
    assert issubclass(technique_class, BaseBalancer)


def test_get_nonexistent_technique(registry):
    """Test attempting to get a non-existent technique"""
    assert registry.get_technique_class("NonExistentTechnique") is None


def test_list_available_techniques(registry):
    """Test listing available techniques"""

    # Register a custom technique
    class CustomBalancer(BaseBalancer):
        def balance(self, X, y):
            return X, y

    registry.register_custom_technique("CustomTechnique", CustomBalancer)

    techniques = registry.list_available_techniques()
    assert isinstance(techniques, dict)
    assert "custom" in techniques
    assert "imblearn" in techniques
    assert "CustomTechnique" in techniques["custom"]


@patch("importlib.import_module")
def test_wrapped_imblearn_technique(
    mock_importlib, mock_imblearn_modules, registry, sample_data
):
    """Test that imblearn techniques are properly wrapped"""

    def mock_import(module_path):
        return mock_imblearn_modules.get(module_path)

    mock_importlib.side_effect = mock_import

    # Force discovery of techniques
    registry._discover_imblearn_techniques()

    # Get wrapped SMOTE
    smote_class = registry.get_technique_class("SMOTE")
    assert smote_class is not None

    # Test the wrapped technique
    smote = smote_class()
    X, y = sample_data
    X_balanced, y_balanced = smote.balance(X, y)

    assert isinstance(X_balanced, np.ndarray)
    assert isinstance(y_balanced, np.ndarray)


def test_error_handling_registration(registry):
    """Test error handling in technique registration"""
    # Test registering None
    with pytest.raises(TypeError):
        registry.register_custom_technique("InvalidTechnique", None)

    # Test registering technique without balance method
    class InvalidBalancer:
        pass

    with pytest.raises(TypeError):
        registry.register_custom_technique("InvalidTechnique", InvalidBalancer)


@patch("importlib.import_module")
def test_import_error_handling(mock_importlib):
    """Test handling of import errors"""
    mock_importlib.side_effect = ImportError("Mock import error")

    # Should not raise an exception, but log a warning
    registry = TechniqueRegistry()
    registry._discover_imblearn_techniques()

    # Registry should still be usable
    assert isinstance(registry.list_available_techniques(), dict)


def test_duplicate_registration(registry):
    """Test registering the same technique name twice"""

    class CustomBalancer1(BaseBalancer):
        def balance(self, X, y):
            return X, y

    class CustomBalancer2(BaseBalancer):
        def balance(self, X, y):
            return X, y

    # Register first technique
    registry.register_custom_technique("CustomTechnique", CustomBalancer1)

    # Register second technique with same name
    registry.register_custom_technique("CustomTechnique", CustomBalancer2)

    # Should use the latest registration
    technique_class = registry.get_technique_class("CustomTechnique")
    assert technique_class == CustomBalancer2
