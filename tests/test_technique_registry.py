import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from balancr import BaseBalancer, TechniqueRegistry


# Mock classes
class MockSMOTE:
    def fit_resample(self, X, y):
        return X, y


class MockRandomUnderSampler:
    def fit_resample(self, X, y):
        return X, y


class MockBaseBalancer(BaseBalancer):
    def balance(self, X, y):
        return X, y


class MockSimpleTechnique(MockBaseBalancer):
    def __init__(self, param1=42, param2="default"):
        super().__init__()
        self.param1 = param1
        self.param2 = param2


class MockTechnique_WithUnderscore(MockBaseBalancer):
    def __init__(self, alpha=0.1, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta


class MockTechnique(MockBaseBalancer):
    def __init__(self, required_param, optional_param=None, complex_default={}):
        super().__init__()
        self.required_param = required_param
        self.optional_param = optional_param
        self.complex_default = complex_default


@pytest.fixture
def registry_with_mocks():
    """Setup a registry with mock techniques for testing"""
    registry = TechniqueRegistry()

    # Add custom techniques to the registry
    registry.custom_techniques = {
        "MockSimpleTechnique": MockSimpleTechnique,
        "MockTechnique_WithUnderscore": MockTechnique_WithUnderscore,
        "MockTechnique": MockTechnique,
    }

    # Mock the imblearn techniques cache
    mock_imblearn_class = MagicMock()
    registry._cached_imblearn_techniques = {
        "SMOTE": ("imblearn.over_sampling", mock_imblearn_class),
        "RandomUnderSampler": ("imblearn.under_sampling", mock_imblearn_class),
    }

    return registry


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


def test_registry_initialisation(registry):
    """Test that registry initialises correctly"""
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


def test_get_technique_class_exact_match(registry_with_mocks):
    """Test getting a technique class with exact name match"""
    # Test custom technique
    technique_class = registry_with_mocks.get_technique_class("MockSimpleTechnique")
    assert technique_class == MockSimpleTechnique

    # Test imblearn technique
    technique_class = registry_with_mocks.get_technique_class("SMOTE")
    assert technique_class is not None


def test_get_technique_class_with_underscore_suffix(registry_with_mocks):
    """Test getting a technique class with underscore suffix"""
    # Test custom technique with suffix
    technique_class = registry_with_mocks.get_technique_class("MockSimpleTechnique_v1")
    assert technique_class == MockSimpleTechnique


def test_get_technique_class_with_dash_suffix(registry_with_mocks):
    """Test getting a technique class with dash suffix"""
    # Test imblearn technique with suffix
    technique_class = registry_with_mocks.get_technique_class("SMOTE-v2")
    assert technique_class is not None


def test_get_technique_class_exact_match_with_underscore(registry_with_mocks):
    """Test that a class with underscore in name is found exactly first"""
    technique_class = registry_with_mocks.get_technique_class(
        "MockTechnique_WithUnderscore"
    )
    assert technique_class == MockTechnique_WithUnderscore


def test_get_technique_class_not_found(registry_with_mocks):
    """Test behavior when technique is not found"""
    technique_class = registry_with_mocks.get_technique_class("NonExistentTechnique")
    assert technique_class is None

    # Test with non-existent base name
    technique_class = registry_with_mocks.get_technique_class("NonExistent_v1")
    assert technique_class is None


def test_get_technique_default_params_exact_match(registry_with_mocks):
    """Test extracting default parameters from a technique with exact name match"""
    params = registry_with_mocks.get_technique_default_params("MockSimpleTechnique")
    assert "param1" in params
    assert params["param1"] == 42
    assert "param2" in params
    assert params["param2"] == "default"


def test_get_technique_default_params_with_suffix(registry_with_mocks):
    """Test extracting default parameters from a technique with suffix"""
    params = registry_with_mocks.get_technique_default_params(
        "MockSimpleTechnique_variant"
    )
    assert "param1" in params
    assert params["param1"] == 42


def test_get_technique_default_params_required_params(registry_with_mocks):
    """Test handling of required parameters (should be None)"""
    params = registry_with_mocks.get_technique_default_params("MockTechnique")
    assert "required_param" in params
    assert params["required_param"] is None
    assert "optional_param" in params
    assert params["optional_param"] is None


def test_extract_params_from_class():
    """Test the _extract_params_from_class method directly"""
    registry = TechniqueRegistry()
    params = registry._extract_params_from_class(MockSimpleTechnique)

    assert "param1" in params
    assert params["param1"] == 42
    assert "param2" in params
    assert params["param2"] == "default"

    # Test with a class that has a required parameter
    params = registry._extract_params_from_class(MockTechnique)
    assert "required_param" in params
    assert params["required_param"] is None  # Should default to None
