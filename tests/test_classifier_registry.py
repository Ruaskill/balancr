import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.base import BaseEstimator
from balancr import ClassifierRegistry


# Mock classes
class MockRandomForestClassifier(BaseEstimator):
    def __init__(self, n_estimators=100, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


# Mock classifier with valid name suffix
class MockRandomForestClassifier_2(BaseEstimator):
    def __init__(self, n_estimators=100, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class MockLogisticRegression(BaseEstimator):
    def __init__(self, C=1.0, max_iter=100):
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


# Non-classifier class (no predict method)
class MockNonClassifier(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self


# Non-estimator class (no BaseEstimator inheritance)
class MockNonEstimator:
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


@pytest.fixture
def mock_sklearn_modules():
    """Mock the sklearn modules for testing"""
    # Mock ensemble module with RandomForestClassifier
    mock_ensemble = MagicMock()
    mock_ensemble.RandomForestClassifier = MockRandomForestClassifier

    # Mock linear_model module with LogisticRegression
    mock_linear_model = MagicMock()
    mock_linear_model.LogisticRegression = MockLogisticRegression

    # Mock other modules
    mock_tree = MagicMock()
    mock_svm = MagicMock()

    return {
        "sklearn.ensemble": mock_ensemble,
        "sklearn.linear_model": mock_linear_model,
        "sklearn.tree": mock_tree,
        "sklearn.svm": mock_svm,
    }


@pytest.fixture
def registry():
    """Create a fresh registry for each test"""
    return ClassifierRegistry()


@pytest.fixture
def sample_data():
    """Create sample data for testing classifier"""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    return X, y


class TestClassifierRegistryInitialisation:
    """Tests for the initialisation of ClassifierRegistry"""

    def test_registry_initialisation(self, registry):
        """Test that registry initialises correctly"""
        assert hasattr(registry, "custom_classifiers")
        assert isinstance(registry.custom_classifiers, dict)
        assert hasattr(registry, "_cached_sklearn_classifiers")
        assert isinstance(registry._cached_sklearn_classifiers, dict)

    @patch("importlib.import_module")
    def test_discover_sklearn_classifiers(
        self, mock_importlib, mock_sklearn_modules, registry
    ):
        """Test discovery of sklearn classifiers on initialisation"""

        def mock_import(module_path):
            return mock_sklearn_modules.get(module_path)

        mock_importlib.side_effect = mock_import

        # Force rediscovery
        registry._discover_sklearn_classifiers()

        # Check if classifiers were discovered
        assert "ensemble" in registry._cached_sklearn_classifiers
        assert (
            "RandomForestClassifier" in registry._cached_sklearn_classifiers["ensemble"]
        )

        assert "linear_model" in registry._cached_sklearn_classifiers
        assert (
            "LogisticRegression" in registry._cached_sklearn_classifiers["linear_model"]
        )


class TestGetClassifierClass:
    """Tests for the get_classifier_class method"""

    def test_get_custom_classifier(self, registry):
        """Test getting a custom classifier"""
        # Register a custom classifier
        registry.custom_classifiers["CustomClassifier"] = MockRandomForestClassifier

        # Get the classifier
        clf_class = registry.get_classifier_class("CustomClassifier")
        assert clf_class is MockRandomForestClassifier

        # Get custom classifier with valid suffix
        clf_class = registry.get_classifier_class("CustomClassifier-2")
        assert clf_class is MockRandomForestClassifier

    @patch("importlib.import_module")
    def test_get_sklearn_classifier(
        self, mock_importlib, mock_sklearn_modules, registry
    ):
        """Test getting a sklearn classifier"""

        def mock_import(module_path):
            return mock_sklearn_modules.get(module_path)

        mock_importlib.side_effect = mock_import

        # Force discovery
        registry._discover_sklearn_classifiers()

        # Test getting a classifier without specifying module
        clf_class = registry.get_classifier_class("RandomForestClassifier")
        assert clf_class == MockRandomForestClassifier

        # Test getting a classifier with module specified
        clf_class = registry.get_classifier_class(
            "LogisticRegression", module_name="linear_model"
        )
        assert clf_class == MockLogisticRegression

        # Test getting a classifier with a valid suffix and with module specified
        clf_class = registry.get_classifier_class(
            "LogisticRegression_2", module_name="linear_model"
        )
        assert clf_class == MockLogisticRegression

        # Test getting a classifier with invalid suffix and with module specified
        clf_class = registry.get_classifier_class(
            "LogisticRegression.BadSuffix", module_name="linear_model"
        )
        assert clf_class is None

    @patch("importlib.import_module")
    def test_get_classifier_with_suffix(
        self, mock_importlib, mock_sklearn_modules, registry
    ):
        """Test getting a classifier whose full name is not in registry, but base name is"""

        def mock_import(module_path):
            return mock_sklearn_modules.get(module_path)

        mock_importlib.side_effect = mock_import

        # Force discovery
        registry._discover_sklearn_classifiers()

        # Test classifier is in registry
        clf_class = registry.get_classifier_class("RandomForestClassifier")
        assert clf_class == MockRandomForestClassifier

        # Test getting a classifier with module specified
        clf_class = registry.get_classifier_class("RandomForestClassifier_2")
        assert clf_class == MockRandomForestClassifier

    @patch("importlib.import_module")
    def test_get_nonexistent_classifier(
        self, mock_importlib, mock_sklearn_modules, registry
    ):
        """Test getting a classifier that doesn't exist"""

        def mock_import(module_path):
            return mock_sklearn_modules.get(module_path)

        mock_importlib.side_effect = mock_import

        # Force discovery
        registry._discover_sklearn_classifiers()

        # Test getting a non-existent classifier
        clf_class = registry.get_classifier_class("NonExistentClassifier")
        assert clf_class is None

        # Test getting a non-existent classifier with module specified
        clf_class = registry.get_classifier_class(
            "NonExistentClassifier", module_name="ensemble"
        )
        assert clf_class is None

    @patch("importlib.import_module")
    def test_rediscover_on_get(self, mock_importlib, mock_sklearn_modules, registry):
        """Test rediscovery of classifiers when one is not found initially"""

        def mock_import(module_path):
            return mock_sklearn_modules.get(module_path)

        mock_importlib.side_effect = mock_import

        # Don't discover initially
        registry._cached_sklearn_classifiers = {}

        # This should trigger _discover_sklearn_classifiers()
        clf_class = registry.get_classifier_class("RandomForestClassifier")
        assert clf_class == MockRandomForestClassifier

        # Same again, but for name with valid suffix
        registry._cached_sklearn_classifiers = {}
        clf_class = registry.get_classifier_class("RandomForestClassifier_2")
        assert clf_class == MockRandomForestClassifier


class TestListAvailableClassifiers:
    """Tests for the list_available_classifiers method"""

    @patch("importlib.import_module")
    def test_list_available_classifiers(
        self, mock_importlib, mock_sklearn_modules, registry
    ):
        """Test listing all available classifiers"""

        def mock_import(module_path):
            return mock_sklearn_modules.get(module_path)

        mock_importlib.side_effect = mock_import

        # Register a custom classifier
        registry.custom_classifiers["CustomClassifier"] = MockRandomForestClassifier

        # Force discovery of sklearn classifiers
        registry._discover_sklearn_classifiers()

        # List all classifiers
        classifiers = registry.list_available_classifiers()

        # Check structure
        assert isinstance(classifiers, dict)
        assert "custom" in classifiers
        assert "sklearn" in classifiers

        # Check custom classifiers
        assert "general" in classifiers["custom"]
        assert "CustomClassifier" in classifiers["custom"]["general"]

        # Check sklearn classifiers
        assert "ensemble" in classifiers["sklearn"]
        assert "RandomForestClassifier" in classifiers["sklearn"]["ensemble"]

        assert "linear_model" in classifiers["sklearn"]
        assert "LogisticRegression" in classifiers["sklearn"]["linear_model"]


class TestRegisterCustomClassifier:
    """Tests for the register_custom_classifier method"""

    def test_register_valid_classifier(self, registry):
        """Test registering a valid custom classifier"""
        registry.register_custom_classifier("CustomRF", MockRandomForestClassifier)

        # Check registration
        assert "CustomRF" in registry.custom_classifiers
        assert registry.custom_classifiers["CustomRF"] == MockRandomForestClassifier

        # Test retrieving the registered classifier
        clf_class = registry.get_classifier_class("CustomRF")
        assert clf_class == MockRandomForestClassifier

    def test_register_invalid_classifier_name(self, registry):
        """Test registering with an invalid name"""
        # Test with empty string
        with pytest.raises(ValueError, match="must be a non-empty string"):
            registry.register_custom_classifier("", MockRandomForestClassifier)

        # Test with None
        with pytest.raises(ValueError, match="must be a non-empty string"):
            registry.register_custom_classifier(None, MockRandomForestClassifier)

    def test_register_none_classifier(self, registry):
        """Test registering None as a classifier"""
        with pytest.raises(TypeError, match="cannot be None"):
            registry.register_custom_classifier("NoneClassifier", None)

    def test_register_non_estimator(self, registry):
        """Test registering a class that doesn't inherit from BaseEstimator"""
        with pytest.raises(
            TypeError, match="must inherit from sklearn.base.BaseEstimator"
        ):
            registry.register_custom_classifier("NonEstimator", MockNonEstimator)

    def test_register_non_classifier(self, registry):
        """Test registering a class that doesn't implement fit and predict"""
        with pytest.raises(
            TypeError, match="must implement 'fit' and 'predict' methods"
        ):
            registry.register_custom_classifier("NonClassifier", MockNonClassifier)

    def test_overwrite_existing_classifier(self, registry):
        """Test overwriting an existing classifier"""
        # Register first classifier
        registry.register_custom_classifier("DuplicateName", MockRandomForestClassifier)

        # Register second classifier with same name
        registry.register_custom_classifier("DuplicateName", MockLogisticRegression)

        # Check the classifier was overwritten
        assert registry.custom_classifiers["DuplicateName"] == MockLogisticRegression


class TestLoadCustomClassifiers:
    """Tests for the _load_custom_classifiers method"""

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.home")
    def test_no_custom_dir(self, mock_home, mock_exists, registry):
        """Test when custom classifiers directory doesn't exist"""
        mock_home.return_value = MagicMock()
        mock_exists.return_value = False

        # Create a fresh registry with empty custom_classifiers
        registry.custom_classifiers = {}

        # Should not raise any errors
        registry._load_custom_classifiers()

        # Custom classifiers should be empty
        assert len(registry.custom_classifiers) == 0

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.home")
    def test_no_metadata_file(self, mock_home, mock_exists, registry):
        """Test when metadata file doesn't exist"""
        mock_home.return_value = MagicMock()

        # Start with empty custom_classifiers
        registry.custom_classifiers = {}

        # Directory exists but metadata file doesn't
        mock_exists.side_effect = lambda path: ".balancr/custom_classifiers" in str(
            path
        ) and "metadata" not in str(path)

        # Should not raise any errors
        registry._load_custom_classifiers()

        # Custom classifiers should be empty
        assert len(registry.custom_classifiers) == 0

    @patch("builtins.open")
    @patch("json.load")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.home")
    def test_invalid_metadata_json(
        self, mock_home, mock_exists, mock_json_load, mock_open, registry
    ):
        """Test handling of invalid JSON in metadata file"""
        mock_home.return_value = MagicMock()
        mock_exists.return_value = True
        mock_json_load.side_effect = Exception("Invalid JSON")

        # Start with empty custom_classifiers
        registry.custom_classifiers = {}

        # Should not raise any errors
        registry._load_custom_classifiers()

        # Custom classifiers should be empty
        assert len(registry.custom_classifiers) == 0

    @patch("builtins.open", new_callable=MagicMock)
    @patch("json.load")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.home")
    @patch("importlib.util.spec_from_file_location")
    @patch("importlib.util.module_from_spec")
    @patch("inspect.getmembers")
    def test_load_valid_classifiers(
        self,
        mock_getmembers,
        mock_module_from_spec,
        mock_spec_from_file,
        mock_home,
        mock_exists,
        mock_json_load,
        mock_open,
        registry,
    ):
        """Test loading valid custom classifiers"""
        # Set up mocks
        mock_home.return_value = MagicMock()
        mock_exists.return_value = True

        # Mock metadata file content
        metadata = {
            "CustomRF": {
                "file": "/path/to/custom_rf.py",
                "class_name": "RandomForestClassifier",
            }
        }
        mock_json_load.return_value = metadata

        # Mock module loading
        mock_spec = MagicMock()
        mock_spec.loader = MagicMock()
        mock_spec_from_file.return_value = mock_spec

        mock_module = MagicMock()
        mock_module_from_spec.return_value = mock_module

        # Mock class in the module
        mock_rf_class = MockRandomForestClassifier
        mock_getmembers.return_value = [("RandomForestClassifier", mock_rf_class)]

        # Call the method
        registry._load_custom_classifiers()

        # Check if classifier was loaded correctly
        assert "CustomRF" in registry.custom_classifiers
        assert registry.custom_classifiers["CustomRF"] == mock_rf_class

    @patch("builtins.open", new_callable=MagicMock)
    @patch("json.load")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.home")
    @patch("importlib.util.spec_from_file_location")
    @patch("importlib.util.module_from_spec")
    @patch("inspect.getmembers")
    @patch("balancr.classifier_registry.logging.warning")
    def test_class_not_found_in_file(
        self,
        mock_warning,
        mock_getmembers,
        mock_module_from_spec,
        mock_spec_from_file,
        mock_home,
        mock_exists,
        mock_json_load,
        mock_open,
        registry,
    ):
        """Test handling of class not found in file"""
        # Set up mocks
        mock_home.return_value = MagicMock()
        mock_exists.return_value = True

        # Mock metadata file content
        metadata = {
            "MissingClassifier": {
                "file": "/path/to/existing.py",
                "class_name": "NonExistentClass",
            }
        }
        mock_json_load.return_value = metadata

        # Mock module loading
        mock_spec = MagicMock()
        mock_spec.loader = MagicMock()
        mock_spec_from_file.return_value = mock_spec

        mock_module = MagicMock()
        mock_module_from_spec.return_value = mock_module

        # Mock different class in the module than what we're looking for
        mock_getmembers.return_value = [("DifferentClass", MockRandomForestClassifier)]

        # Call the method
        registry._load_custom_classifiers()

        # Check warning was logged
        mock_warning.assert_called_once()
        assert "Class NonExistentClass not found" in mock_warning.call_args[0][0]

        # Classifier should not be loaded
        assert "MissingClassifier" not in registry.custom_classifiers

    @patch("builtins.open", new_callable=MagicMock)
    @patch("json.load")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.home")
    @patch("importlib.util.spec_from_file_location")
    @patch("balancr.classifier_registry.logging.warning")
    def test_module_loading_error(
        self,
        mock_warning,
        mock_spec_from_file,
        mock_home,
        mock_exists,
        mock_json_load,
        mock_open,
        registry,
    ):
        """Test handling of module loading errors"""
        # Set up mocks
        mock_home.return_value = MagicMock()
        mock_exists.return_value = True

        # Mock metadata file content
        metadata = {
            "ErrorClassifier": {"file": "/path/to/error.py", "class_name": "ErrorClass"}
        }
        mock_json_load.return_value = metadata

        # Mock spec_from_file_location to return None (error condition)
        mock_spec_from_file.return_value = None

        # Call the method
        registry._load_custom_classifiers()

        # Check warning was logged
        mock_warning.assert_called_once()
        assert "Could not load module" in mock_warning.call_args[0][0]

        # Classifier should not be loaded
        assert "ErrorClassifier" not in registry.custom_classifiers


class TestGetSklearnClassifiers:
    """Tests for the _get_sklearn_classifiers_by_module method"""

    @patch("importlib.import_module")
    def test_get_sklearn_classifiers_by_module(
        self, mock_importlib, mock_sklearn_modules, registry
    ):
        """Test getting sklearn classifiers organised by module"""

        def mock_import(module_path):
            return mock_sklearn_modules.get(module_path)

        mock_importlib.side_effect = mock_import

        # Force discovery
        registry._discover_sklearn_classifiers()

        # Get classifiers by module
        classifiers_by_module = registry._get_sklearn_classifiers_by_module()

        # Check structure
        assert isinstance(classifiers_by_module, dict)
        assert "ensemble" in classifiers_by_module
        assert "linear_model" in classifiers_by_module

        # Check contents
        assert "RandomForestClassifier" in classifiers_by_module["ensemble"]
        assert "LogisticRegression" in classifiers_by_module["linear_model"]


class TestErrorHandling:
    """Tests for error handling in ClassifierRegistry"""

    @patch("importlib.import_module")
    def test_import_error_handling(self, mock_importlib, registry):
        """Test handling of import errors during discovery"""
        mock_importlib.side_effect = ImportError("Module not found")

        # Should not raise an exception
        registry._discover_sklearn_classifiers()

        # Registry should still be usable
        assert isinstance(registry.list_available_classifiers(), dict)


class TestIntegrationTests:
    """Integration tests for ClassifierRegistry"""

    @patch("importlib.import_module")
    def test_full_workflow(
        self, mock_importlib, mock_sklearn_modules, registry, sample_data
    ):
        """Test a full workflow of registering, discovering, and using classifiers"""

        def mock_import(module_path):
            return mock_sklearn_modules.get(module_path)

        mock_importlib.side_effect = mock_import

        # 1. Register a custom classifier
        registry.register_custom_classifier("MyCustomRF", MockRandomForestClassifier)

        # 2. Discover sklearn classifiers
        registry._discover_sklearn_classifiers()

        # 3. List all available classifiers
        classifiers = registry.list_available_classifiers()
        assert "custom" in classifiers
        assert "sklearn" in classifiers

        # 4. Get and use the custom classifier
        custom_clf_class = registry.get_classifier_class("MyCustomRF")
        assert custom_clf_class is not None

        custom_clf = custom_clf_class(n_estimators=10)
        X, y = sample_data
        custom_clf.fit(X, y)
        predictions = custom_clf.predict(X)
        assert len(predictions) == len(X)

        # 5. Get and use a sklearn classifier
        sklearn_clf_class = registry.get_classifier_class("LogisticRegression")
        assert sklearn_clf_class is not None

        sklearn_clf = sklearn_clf_class(C=0.5)
        sklearn_clf.fit(X, y)
        predictions = sklearn_clf.predict(X)
        assert len(predictions) == len(X)
