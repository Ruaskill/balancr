"""Tests for classifier-related commands in the balancr CLI."""

import json
from pathlib import Path
import pytest
from unittest.mock import mock_open, patch, MagicMock

from balancr.cli import commands
from sklearn.base import BaseEstimator


@pytest.fixture
def mock_config_path(tmp_path):
    """Create a temporary config file path for testing."""
    return tmp_path / "test_config.json"


@pytest.fixture
def mock_registry():
    """Mock the ClassifierRegistry class."""
    mock = MagicMock()
    mock.list_available_classifiers.return_value = {
        "custom": {"general": ["CustomClassifier1", "CustomClassifier2"]},
        "sklearn": {
            "ensemble": ["RandomForestClassifier", "GradientBoostingClassifier"],
            "linear_model": ["LogisticRegression"],
        },
    }
    return mock


@pytest.fixture
def args_select_classifier(mock_config_path):
    """Create mock arguments for select_classifier command."""
    args = MagicMock()
    args.classifiers = ["RandomForestClassifier", "LogisticRegression"]
    args.list_available = False
    args.append = False
    args.config_path = str(mock_config_path)
    args.verbose = False
    return args


class TestSelectClassifierCommand:
    """Tests for the select_classifier command."""

    @patch("balancr.cli.commands.ClassifierRegistry")
    def test_list_available_classifiers(
        self, mock_registry_class, args_select_classifier, mock_registry
    ):
        """Test listing available classifiers."""
        # Set up for listing
        args_select_classifier.list_available = True
        args_select_classifier.classifiers = []

        # Set up mock registry
        mock_registry_class.return_value = mock_registry

        # Mock print to capture output
        with patch("builtins.print") as mock_print:
            # Call function
            result = commands.select_classifier(args_select_classifier)

            # Verify registry was used to list classifiers
            mock_registry_class.assert_called_once()
            mock_registry.list_available_classifiers.assert_called_once()

            # Verify output includes both custom and sklearn classifiers
            output = " ".join(str(call[0][0]) for call in mock_print.call_args_list)
            assert "Custom Classifiers" in output
            assert "Scikit-learn Classifiers" in output
            assert "RandomForestClassifier" in output
            assert "LogisticRegression" in output

            # Verify result
            assert result == 0

    def test_list_available_classifiers_no_registry(self, args_select_classifier):
        """Test list_available_classifiers when ClassifierRegistry is not available."""
        # Set up for listing
        args_select_classifier.list_available = True
        args_select_classifier.classifiers = []

        # Mock ClassifierRegistry to be None
        with patch("balancr.cli.commands.ClassifierRegistry", None), patch(
            "balancr.cli.commands.logging.error"
        ) as mock_error:

            # Call the function
            result = commands.list_available_classifiers(args_select_classifier)

            # Verify error was logged
            mock_error.assert_called_once()
            assert "Classifier registry not available" in mock_error.call_args[0][0]

            # Verify result
            assert result == 1

    @patch("balancr.cli.commands.ClassifierRegistry")
    @patch("balancr.cli.commands.get_classifier_default_params")
    @patch("balancr.cli.config.load_config")
    def test_select_classifiers_success(
        self,
        mock_load_config,
        mock_get_params,
        mock_registry_class,
        args_select_classifier,
        mock_registry,
    ):
        """Test successful selection of classifiers."""
        # Set up mock registry
        mock_registry_class.return_value = mock_registry

        # Mock get_classifier_default_params to return some params
        mock_get_params.return_value = {"n_estimators": 100, "random_state": 42}

        # Set up mock classifier classes
        mock_rf_class = MagicMock()
        mock_lr_class = MagicMock()

        # Configure registry to return classifiers
        mock_registry.get_classifier_class.side_effect = [mock_rf_class, mock_lr_class]

        # Set up config loading
        mock_load_config.return_value = {}

        # Mock open and json.dump to capture the final configuration
        m_json_dump = MagicMock()

        with patch("builtins.open", mock_open()) as m_file, patch(
            "json.dump", m_json_dump
        ):
            # Call function
            result = commands.select_classifier(args_select_classifier)

            # Verify registry was used to validate classifiers
            mock_registry_class.assert_called_once()
            assert mock_registry.get_classifier_class.call_count == 2

            # Verify file was opened and json was written
            m_file.assert_called_once()
            m_json_dump.assert_called_once()

            # Verify the configuration that would be written
            config_arg = m_json_dump.call_args[0][0]
            assert "classifiers" in config_arg
            assert "RandomForestClassifier" in config_arg["classifiers"]
            assert "LogisticRegression" in config_arg["classifiers"]

            # Verify each classifier has parameters
            assert "n_estimators" in config_arg["classifiers"]["RandomForestClassifier"]

            # Verify result
            assert result == 0

    @patch("balancr.cli.commands.ClassifierRegistry")
    @patch("balancr.cli.commands.get_classifier_default_params")
    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.config.update_config")
    def test_select_classifiers_append(
        self,
        mock_update_config,
        mock_load_config,
        mock_get_params,
        mock_registry_class,
        args_select_classifier,
        mock_registry,
    ):
        """Test appending classifiers to existing configuration."""
        # Set up for appending
        args_select_classifier.append = True

        # Set up mock registry
        mock_registry_class.return_value = mock_registry

        # Mock get_classifier_default_params to return some params
        mock_get_params.return_value = {"n_estimators": 100, "random_state": 42}

        # Set up mock classifier classes
        mock_rf_class = MagicMock()
        mock_lr_class = MagicMock()

        # Configure registry to return classifiers
        mock_registry.get_classifier_class.side_effect = [mock_rf_class, mock_lr_class]

        # Set up existing config with some classifiers
        mock_load_config.return_value = {
            "classifiers": {"GradientBoostingClassifier": {"n_estimators": 50}}
        }

        # Call function
        result = commands.select_classifier(args_select_classifier)

        # Verify config was updated (not replaced)
        mock_update_config.assert_called_once()
        settings = mock_update_config.call_args[0][1]

        # Expected result should include both old and new classifiers
        assert "classifiers" in settings
        classifiers = settings["classifiers"]
        assert "GradientBoostingClassifier" in classifiers  # Existing classifier
        assert "RandomForestClassifier" in classifiers  # New classifier
        assert "LogisticRegression" in classifiers  # New classifier

        # Verify result
        assert result == 0

    @patch("balancr.cli.commands.ClassifierRegistry")
    @patch("balancr.cli.commands.logging.error")
    def test_select_invalid_classifiers(
        self, mock_error, mock_registry_class, args_select_classifier
    ):
        """Test handling of invalid classifier selection."""
        # Set up mock registry with limited classifiers
        mock_registry = MagicMock()
        mock_registry.get_classifier_class.return_value = None
        mock_registry_class.return_value = mock_registry

        # Request invalid classifiers
        args_select_classifier.classifiers = ["InvalidClassifier"]

        # Call function
        result = commands.select_classifier(args_select_classifier)

        # Verify both error messages were logged
        assert mock_error.call_count == 2
        assert any(
            "Classifier 'InvalidClassifier' not found" in call[0][0]
            for call in mock_error.call_args_list
        )
        assert any(
            "No valid classifiers selected" in call[0][0]
            for call in mock_error.call_args_list
        )

        # Verify result
        assert result == 1

    @patch("balancr.cli.commands.ClassifierRegistry", None)
    @patch("balancr.cli.commands.logging.error")
    def test_select_classifiers_no_registry(self, mock_error, args_select_classifier):
        """Test classifier selection when registry is not available."""
        # Call function
        result = commands.select_classifier(args_select_classifier)

        # Verify error was logged
        mock_error.assert_called_once()
        assert "Classifier registry not available" in mock_error.call_args[0][0]

        # Verify result
        assert result == 1

    @patch("balancr.cli.commands.ClassifierRegistry")
    @patch("balancr.cli.commands.get_classifier_default_params")
    @patch("balancr.cli.config.load_config")
    def test_select_classifiers_config_error(
        self,
        mock_load_config,
        mock_get_params,
        mock_registry_class,
        args_select_classifier,
        mock_registry,
    ):
        """Test error handling in select_classifier during config operations."""
        # Set up mock registry with valid classifiers
        mock_registry_class.return_value = mock_registry
        mock_registry.get_classifier_class.return_value = MagicMock()

        # Mock get_classifier_default_params to return some params
        mock_get_params.return_value = {"param": "value"}

        # Make load_config raise an exception
        mock_load_config.side_effect = Exception("Config error")

        # Mock logging for error capture
        with patch("balancr.cli.commands.logging.error") as mock_error:
            # Call function
            result = commands.select_classifier(args_select_classifier)

            # Verify error was logged
            mock_error.assert_called_once()
            assert "Failed to select classifiers" in mock_error.call_args[0][0]

            # Verify result
            assert result == 1


class TestGetClassifierDefaultParams:
    """Tests for the get_classifier_default_params function."""

    def test_extract_basic_params(self):
        """Test extracting parameters from a classifier with basic types."""

        # Create a simple classifier class with basic parameter types
        class SimpleClassifier:
            def __init__(self, param_int=42, param_float=3.14, param_str="default"):
                self.param_int = param_int
                self.param_float = param_float
                self.param_str = param_str

        # Extract parameters
        params = commands.get_classifier_default_params(SimpleClassifier)

        # Verify extracted parameters
        assert params["param_int"] == 42
        assert params["param_float"] == 3.14
        assert params["param_str"] == "default"

    def test_extract_complex_params(self):
        """Test handling of complex parameter types that need string conversion."""

        # Create a classifier with non-JSON serialisable parameters
        class ComplexClassifier:
            def __init__(self, complex_param=complex(1, 2), callable_param=print):
                self.complex_param = complex_param
                self.callable_param = callable_param

        # Extract parameters
        params = commands.get_classifier_default_params(ComplexClassifier)

        # Complex types should be converted to strings
        assert isinstance(params["complex_param"], str)
        assert isinstance(params["callable_param"], str)
        assert (
            "(1+2j)" in params["complex_param"]
        )  # String representation of complex(1,2)

    def test_extract_collection_params(self):
        """Test parameters that are collections (list, dict)."""

        # Create a classifier with collection parameters
        class CollectionClassifier:
            def __init__(self, list_param=[1, 2, 3], dict_param={"a": 1, "b": 2}):
                self.list_param = list_param
                self.dict_param = dict_param

        # Extract parameters
        params = commands.get_classifier_default_params(CollectionClassifier)

        # Collection types should be preserved
        assert params["list_param"] == [1, 2, 3]
        assert params["dict_param"] == {"a": 1, "b": 2}

    def test_extract_none_params(self):
        """Test parameters with None default value."""

        # Create a classifier with None parameters
        class NoneClassifier:
            def __init__(self, none_param=None):
                self.none_param = none_param

        # Extract parameters
        params = commands.get_classifier_default_params(NoneClassifier)

        # None should be preserved
        assert params["none_param"] is None

    def test_extract_params_no_defaults(self):
        """Test parameters without default values."""

        # Create a classifier with parameters that don't have defaults
        class NoDefaultsClassifier:
            def __init__(self, required_param, optional_param=True):
                self.required_param = required_param
                self.optional_param = optional_param

        # Extract parameters
        params = commands.get_classifier_default_params(NoDefaultsClassifier)

        # Parameters without defaults should be None
        assert params["required_param"] is None
        # Parameters with defaults should retain their values
        assert params["optional_param"] is True

    @patch("balancr.cli.commands.logging.warning")
    def test_extract_params_exception_with_name(self, mock_warning):
        """Test handling of exceptions during parameter extraction with a named class."""

        # Create a class that will raise an exception when inspect.signature is called
        class ProblematicClassifier:
            # Override __init__ to make it not inspectable
            __init__ = None

        # Call function
        commands.get_classifier_default_params(ProblematicClassifier)

        # Should log warning
        mock_warning.assert_called_once()
        assert (
            "Error extracting parameters from ProblematicClassifier"
            in mock_warning.call_args[0][0]
        )


class TestRegisterClassifiersCommand:
    """Tests for the register_classifiers command."""

    @pytest.fixture
    def mock_classifier_file(self, tmp_path):
        """Create a mock Python file with a classifier class."""
        classifier_file = tmp_path / "mock_classifier.py"
        with open(classifier_file, "w") as f:
            f.write(
                """
from sklearn.base import BaseEstimator

class MockClassifier(BaseEstimator):
    def __init__(self, param1=10, param2="default"):
        self.param1 = param1
        self.param2 = param2

    def fit(self, X, y):
        return self

    def predict(self, X):
        return [0] * len(X)
"""
            )
        return classifier_file

    @pytest.fixture
    def args_register_classifiers(self, mock_config_path, mock_classifier_file):
        """Create mock arguments for register_classifiers command."""
        args = MagicMock()
        args.file_path = str(mock_classifier_file)
        args.folder_path = None
        args.name = None
        args.class_name = None
        args.overwrite = False
        args.remove = None
        args.remove_all = False
        args.config_path = str(mock_config_path)
        args.verbose = False
        return args

    @patch("balancr.cli.commands.ClassifierRegistry")
    @patch("balancr.cli.commands._register_classifier_from_file")
    def test_register_classifier_from_file(
        self,
        mock_register_from_file,
        mock_registry_class,
        args_register_classifiers,
        mock_registry,
    ):
        """Test registering a classifier from a file."""
        # Set up mocks
        mock_registry_class.return_value = mock_registry
        mock_register_from_file.return_value = [
            "MockClassifier"
        ]  # Simulate successful registration

        # Mock Path.exists to return True for the file
        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_file", return_value=True
        ), patch("pathlib.Path.suffix", ".py", create=True):

            # Call function
            result = commands.register_classifiers(args_register_classifiers)

            # Verify _register_classifier_from_file was called
            mock_register_from_file.assert_called_once_with(
                mock_registry,
                Path(args_register_classifiers.file_path),
                args_register_classifiers.name,
                args_register_classifiers.class_name,
                args_register_classifiers.overwrite,
            )

            # Verify result
            assert result == 0

    @patch("balancr.cli.commands.ClassifierRegistry")
    @patch("balancr.cli.commands._register_classifier_from_file")
    def test_register_classifier_with_custom_name(
        self,
        mock_register_from_file,
        mock_registry_class,
        args_register_classifiers,
        mock_registry,
    ):
        """Test registering a classifier with a custom name."""
        # Set up mocks
        mock_registry_class.return_value = mock_registry
        mock_register_from_file.return_value = [
            "MyCustomClassifier"
        ]  # Simulate successful registration

        # Set custom name and class name
        args_register_classifiers.name = "MyCustomClassifier"
        args_register_classifiers.class_name = "MockClassifier"

        # Mock Path.exists to return True for the file
        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_file", return_value=True
        ), patch("pathlib.Path.suffix", ".py", create=True):

            # Call function
            result = commands.register_classifiers(args_register_classifiers)

            # Verify _register_classifier_from_file was called with custom name and class
            mock_register_from_file.assert_called_once_with(
                mock_registry,
                Path(args_register_classifiers.file_path),
                "MyCustomClassifier",  # Custom name
                "MockClassifier",  # Class name
                args_register_classifiers.overwrite,
            )

            # Verify result
            assert result == 0

    @patch("balancr.cli.commands.ClassifierRegistry")
    def test_register_classifier_from_folder(
        self, mock_registry_class, args_register_classifiers, mock_registry, tmp_path
    ):
        """Test registering classifiers from a folder."""
        mock_registry_class.return_value = mock_registry

        # Create folder with Python files
        folder_path = tmp_path / "classifiers"
        folder_path.mkdir()

        # Create a couple of Python files in the folder
        for i in range(2):
            with open(folder_path / f"classifier{i}.py", "w") as f:
                f.write(
                    f"""
from sklearn.base import BaseEstimator

class MockClassifier{i}(BaseEstimator):
    def __init__(self, param1=10):
        self.param1 = param1

    def fit(self, X, y):
        return self

    def predict(self, X):
        return [0] * len(X)
"""
                )

        # Set up args for folder registration
        args_register_classifiers.file_path = None
        args_register_classifiers.folder_path = str(folder_path)

        # Mock Path.glob to return our Python files
        with patch.object(Path, "glob") as mock_glob:
            mock_glob.return_value = [
                folder_path / f"classifier{i}.py" for i in range(2)
            ]

            # Mock _register_classifier_from_file to return success
            with patch(
                "balancr.cli.commands._register_classifier_from_file"
            ) as mock_register_from_file:
                mock_register_from_file.return_value = [
                    f"MockClassifier{i}" for i in range(2)
                ]

                # Mock path checks
                with patch("pathlib.Path.exists", return_value=True), patch(
                    "pathlib.Path.is_dir", return_value=True
                ):

                    # Call function
                    result = commands.register_classifiers(args_register_classifiers)

                    # Verify glob was called to find Python files
                    mock_glob.assert_called_once()

                    # Verify _register_classifier_from_file was called for each file
                    assert mock_register_from_file.call_count == 2

                    # Verify result
                    assert result == 0

    def test_register_classifiers_remove_path(self, args_register_classifiers):
        """Test register_classifiers with remove option."""
        # Set up for removal
        args_register_classifiers.file_path = None
        args_register_classifiers.remove = ["CustomClassifier1"]

        # Mock _remove_classifiers to return a known value
        with patch(
            "balancr.cli.commands._remove_classifiers"
        ) as mock_remove:
            mock_remove.return_value = 42  # arbitrary return value

            # Call function
            result = commands.register_classifiers(args_register_classifiers)

            # Verify _remove_classifiers was called and its result returned
            mock_remove.assert_called_once_with(args_register_classifiers)
            assert result == 42

    def test_register_classifiers_remove_all_path(self, args_register_classifiers):
        """Test register_classifiers with remove_all option."""
        # Set up for removal
        args_register_classifiers.file_path = None
        args_register_classifiers.remove_all = True

        # Mock _remove_classifiers to return a known value
        with patch(
            "balancr.cli.commands._remove_classifiers"
        ) as mock_remove:
            mock_remove.return_value = 0

            # Call function
            result = commands.register_classifiers(args_register_classifiers)

            # Verify _remove_classifiers was called and its result returned
            mock_remove.assert_called_once_with(args_register_classifiers)
            assert result == 0

    def test_register_classifier_not_python_file(self, args_register_classifiers):
        """Test handling when file is not a Python file."""
        # Mock file path to exist but not be a Python file
        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_file", return_value=True
        ), patch("pathlib.Path.suffix", ".txt", create=True), patch(
            "balancr.cli.commands.logging.error"
        ) as mock_error:

            # Call function
            result = commands.register_classifiers(args_register_classifiers)

            # Verify error was logged
            mock_error.assert_called_once()
            assert "Not a Python file" in mock_error.call_args[0][0]

            # Verify result
            assert result == 1

    def test_register_classifier_folder_not_found(self, args_register_classifiers):
        """Test handling when folder path doesn't exist."""
        # Set up for folder registration with non-existent path
        args_register_classifiers.file_path = None
        args_register_classifiers.folder_path = "/nonexistent/folder"

        # Mock folder path to not exist
        with patch("pathlib.Path.exists", return_value=False), patch(
            "balancr.cli.commands.logging.error"
        ) as mock_error:

            # Call function
            result = commands.register_classifiers(args_register_classifiers)

            # Verify error was logged
            mock_error.assert_called_once()
            assert "Folder not found" in mock_error.call_args[0][0]

            # Verify result
            assert result == 1

    def test_register_classifier_not_a_directory(self, args_register_classifiers):
        """Test handling when path is not a directory."""
        # Set up for folder registration with path that's not a directory
        args_register_classifiers.file_path = None
        args_register_classifiers.folder_path = "/path/to/file.txt"

        # Mock path to exist but not be a directory
        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("balancr.cli.commands.logging.error") as mock_error:

            # Call function
            result = commands.register_classifiers(args_register_classifiers)

            # Verify error was logged
            mock_error.assert_called_once()
            assert "Not a directory" in mock_error.call_args[0][0]

            # Verify result
            assert result == 1

    def test_register_classifier_no_valid_classifiers_found(
        self, args_register_classifiers, mock_registry
    ):
        """Test handling when no valid classifiers are found."""
        # Set up for folder registration
        args_register_classifiers.file_path = None
        args_register_classifiers.folder_path = "/valid/folder"

        # Mock path checks and registry
        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=True
        ), patch("pathlib.Path.glob") as mock_glob, patch(
            "balancr.cli.commands.ClassifierRegistry"
        ) as mock_registry_class, patch(
            "balancr.cli.commands._register_classifier_from_file"
        ) as mock_register_from_file, patch(
            "balancr.cli.commands.logging.warning"
        ) as mock_warning:

            # Set up mock registry
            mock_registry_class.return_value = mock_registry

            # Set up mock glob to return one file
            mock_glob.return_value = [Path("/valid/folder/classifier.py")]

            # _register_classifier_from_file returns empty list - no classifiers found
            mock_register_from_file.return_value = []

            # Call function
            result = commands.register_classifiers(args_register_classifiers)

            # Verify warning was logged
            mock_warning.assert_called_once()
            assert "No valid classifiers found" in mock_warning.call_args[0][0]

            # Verify result
            assert result == 1

    def test_register_classifiers_exception(self, args_register_classifiers):
        """Test handling of exceptions in register_classifiers."""
        # Make Path.exists raise an exception
        with patch("pathlib.Path.exists", side_effect=Exception("Test error")), patch(
            "balancr.cli.commands.logging.error"
        ) as mock_error, patch("traceback.print_exc") as mock_traceback:

            # Set up verbose mode
            args_register_classifiers.verbose = True

            # Call function
            result = commands.register_classifiers(args_register_classifiers)

            # Verify error was logged
            mock_error.assert_called_once()
            assert "Error registering classifiers" in mock_error.call_args[0][0]

            # Verify traceback was printed due to verbose mode
            mock_traceback.assert_called_once()

            # Verify result
            assert result == 1

    @patch("balancr.cli.commands.ClassifierRegistry")
    def test_register_classifier_file_not_found(
        self, mock_registry_class, args_register_classifiers, mock_registry
    ):
        """Test handling when classifier file doesn't exist."""
        mock_registry_class.return_value = mock_registry

        # Set a non-existent file path
        args_register_classifiers.file_path = "nonexistent_file.py"

        # Mock Path.exists to return False
        with patch("pathlib.Path.exists", return_value=False), patch(
            "balancr.cli.commands.logging.error"
        ) as mock_error:

            # Call function
            result = commands.register_classifiers(args_register_classifiers)

            # Verify error was logged
            mock_error.assert_called_once()
            assert "File not found" in mock_error.call_args[0][0]

            # Verify result
            assert result == 1

    @patch("balancr.cli.commands.ClassifierRegistry", None)
    def test_register_classifiers_no_registry(self, args_register_classifiers):
        """Test handling when ClassifierRegistry is not available."""
        # Call function
        result = commands.register_classifiers(args_register_classifiers)

        # Verify result
        assert result == 1


class TestRegisterClassifierFromFile:
    """Tests for the _register_classifier_from_file helper function."""

    @pytest.fixture
    def mock_module(self):
        """Create a mock module with classifier classes."""
        mock = MagicMock()

        # Add two classifier classes to the module
        class Classifier1(BaseEstimator):
            def __init__(self, param1=10):
                self.param1 = param1

            def fit(self, X, y):
                return self

            def predict(self, X):
                return [0] * len(X)

        class Classifier2(BaseEstimator):
            def __init__(self, param1=20):
                self.param1 = param1

            def fit(self, X, y):
                return self

            def predict(self, X):
                return [1] * len(X)

        # Set module name property
        mock.__name__ = "mock_module"

        # Add classes to the module mock
        mock.Classifier1 = Classifier1
        mock.Classifier2 = Classifier2

        # Configure inspect.getmembers to return our classes
        def mock_getmembers(module, predicate):
            return [("Classifier1", Classifier1), ("Classifier2", Classifier2)]

        return mock, mock_getmembers

    @patch("os.makedirs")
    @patch("importlib.util.spec_from_file_location")
    @patch("importlib.util.module_from_spec")
    @patch("inspect.getmembers")
    @patch("inspect.isclass")
    @patch("balancr.cli.commands.logging.warning")
    def test_register_from_file_no_valid_classes(
        self,
        mock_logging_warning,
        mock_isclass,
        mock_getmembers,
        mock_module_from_spec,
        mock_spec_from_file,
        mock_makedirs,
    ):
        """Test _register_classifier_from_file when no valid classifier classes are found."""
        # Set up mocks
        mock_registry = MagicMock()
        mock_file_path = Path("/path/to/classifier_file.py")

        # Create a module with no classifier classes
        module_mock = MagicMock()
        module_mock.__name__ = "mock_module"

        # Configure module loading mocks
        mock_spec = MagicMock()
        mock_spec.loader = MagicMock()
        mock_spec_from_file.return_value = mock_spec
        mock_module_from_spec.return_value = module_mock

        # Configure inspection mocks to return no classes
        mock_getmembers.return_value = []
        mock_isclass.return_value = True

        # Call function
        result = commands._register_classifier_from_file(
            mock_registry, mock_file_path, None, None, False
        )

        # Verify warning was logged
        mock_logging_warning.assert_called_once()
        assert (
            "No valid classifier classes found" in mock_logging_warning.call_args[0][0]
        )

        # Verify no classifiers were registered
        assert len(result) == 0

    def test_register_from_file_class_name_not_found(self, tmp_path):
        """Test when a specified class name is not found in the file."""
        # Create a temporary directory structure
        custom_dir = tmp_path / ".balancr" / "custom_classifiers"
        custom_dir.mkdir(parents=True, exist_ok=True)

        # Create a real Python file with an actual classifier class
        classifier_file = tmp_path / "single_classifier.py"
        code = """from sklearn.base import BaseEstimator

class ExistingClassifier(BaseEstimator):
    def __init__(self, param1=42):
        self.param1 = param1

    def fit(self, X, y):
        return self

    def predict(self, X):
        return [0] * len(X)
    """

        with open(classifier_file, "w") as f:
            f.write(code)

        # Create a mock registry
        mock_registry = MagicMock()

        # Mock Path.home to return our temporary directory
        with patch("pathlib.Path.home", return_value=tmp_path), patch(
            "balancr.cli.commands.logging.error"
        ) as mock_error:
            # Call function with a non-existent class name
            result = commands._register_classifier_from_file(
                mock_registry, classifier_file, None, "NonExistentClass", False
            )

            # Verify the error was logged with the correct message
            mock_error.assert_called_once()
            error_msg = mock_error.call_args[0][0]
            assert "Class 'NonExistentClass' not found" in error_msg

            # Verify no classifiers were registered
            assert result == []

    def test_register_from_file_custom_name_without_class_specified(self, tmp_path):
        """Test when a custom name is provided but multiple classes exist without specifying class_name."""
        # Create a temporary directory structure
        custom_dir = tmp_path / ".balancr" / "custom_classifiers"
        custom_dir.mkdir(parents=True, exist_ok=True)

        # Create a real Python file with multiple classifier classes
        classifier_file = tmp_path / "multi_classifier.py"
        code = """from sklearn.base import BaseEstimator

class ClassifierOne(BaseEstimator):
    def __init__(self, param1=42):
        self.param1 = param1

    def fit(self, X, y):
        return self

    def predict(self, X):
        return [0] * len(X)

class ClassifierTwo(BaseEstimator):
    def __init__(self, param1=42):
        self.param1 = param1

    def fit(self, X, y):
        return self

    def predict(self, X):
        return [1] * len(X)
"""

        with open(classifier_file, "w") as f:
            f.write(code)

        # Create a mock registry
        mock_registry = MagicMock()

        # Mock Path.home to return our temporary directory
        with patch("pathlib.Path.home", return_value=tmp_path), patch(
            "balancr.cli.commands.logging.error"
        ) as mock_error:
            # Call function with a custom name but no class specified
            result = commands._register_classifier_from_file(
                mock_registry, classifier_file, "CustomName", None, False
            )

            # Verify the error was logged
            mock_error.assert_called_once()
            error_msg = mock_error.call_args[0][0]
            assert "Multiple classifier classes found" in error_msg
            assert "custom name provided" in error_msg
            assert "--class-name" in error_msg

            # Verify no classifiers were registered
            assert result == []

    @patch("os.makedirs")
    @patch("importlib.util.spec_from_file_location")
    @patch("balancr.cli.commands.logging.error")
    def test_register_from_file_module_loading_error(
        self, mock_logging_error, mock_spec_from_file, mock_makedirs
    ):
        """Test _register_classifier_from_file when there's an error loading the module."""
        # Set up mocks
        mock_registry = MagicMock()
        mock_file_path = Path("/path/to/classifier_file.py")

        # Make spec_from_file_location return None (error condition)
        mock_spec_from_file.return_value = None

        # Call function
        result = commands._register_classifier_from_file(
            mock_registry, mock_file_path, None, None, False
        )

        # Verify error was logged
        mock_logging_error.assert_called_once()
        assert "Could not load module" in mock_logging_error.call_args[0][0]

        # Verify no classifiers were registered
        assert len(result) == 0

    def test_register_from_file_with_custom_name_and_class_name(self, tmp_path):
        """Test registering a classifier with a custom name for a specified class."""
        # Create a temporary directory structure
        custom_dir = tmp_path / ".balancr" / "custom_classifiers"
        custom_dir.mkdir(parents=True, exist_ok=True)

        # Create a real Python file with multiple classifier classes
        classifier_file = tmp_path / "multi_classifier.py"
        code = """from sklearn.base import BaseEstimator

class ClassifierOne(BaseEstimator):
    def __init__(self, param1=42):
        self.param1 = param1

    def fit(self, X, y):
        return self

    def predict(self, X):
        return [0] * len(X)

class ClassifierTwo(BaseEstimator):
    def __init__(self, param1=42):
        self.param1 = param1

    def fit(self, X, y):
        return self

    def predict(self, X):
        return [1] * len(X)
    """

        with open(classifier_file, "w") as f:
            f.write(code)

        # Create a mock registry
        mock_registry = MagicMock()
        mock_registry.list_available_classifiers.return_value = {
            "custom": {"general": []},
            "sklearn": {},
        }

        # Mock Path.home to return our temporary directory
        with patch("pathlib.Path.home", return_value=tmp_path), patch(
            "balancr.cli.commands.datetime"
        ) as mock_datetime, patch(
            "balancr.cli.commands.logging.info"
        ) as mock_info:

            # Set a fixed datetime for reproducibility
            mock_datetime.now.return_value.isoformat.return_value = (
                "2023-01-01T00:00:00"
            )

            # Call function with a custom name and specific class
            result = commands._register_classifier_from_file(
                mock_registry, classifier_file, "CustomName", "ClassifierOne", False
            )

            # Verify classifier was registered with custom name
            mock_registry.register_custom_classifier.assert_called_once()
            args = mock_registry.register_custom_classifier.call_args[0]
            assert args[0] == "CustomName"  # Custom name was used
            assert args[1].__name__ == "ClassifierOne"  # Correct class was selected

            # Verify success message was logged
            mock_info.assert_any_call("Successfully registered classifier: CustomName")

            # Verify result contains the registered classifier with custom name
            assert "CustomName" in result

    def test_register_from_file_classifier_already_exists(self, tmp_path):
        """Test when classifier already exists and overwrite is False."""
        # Create a temporary directory structure
        custom_dir = tmp_path / ".balancr" / "custom_classifiers"
        custom_dir.mkdir(parents=True, exist_ok=True)

        # Create a real Python file with a classifier class
        classifier_file = tmp_path / "existing_classifier.py"
        code = """from sklearn.base import BaseEstimator

class ExistingClassifier(BaseEstimator):
    def __init__(self, param1=42):
        self.param1 = param1

    def fit(self, X, y):
        return self

    def predict(self, X):
        return [0] * len(X)
    """

        with open(classifier_file, "w") as f:
            f.write(code)

        # Create a mock registry that reports the classifier already exists
        mock_registry = MagicMock()
        mock_registry.list_available_classifiers.return_value = {
            "custom": {"general": ["ExistingClassifier"]},
            "sklearn": {},
        }

        # Mock Path.home to return our temporary directory
        with patch("pathlib.Path.home", return_value=tmp_path), patch(
            "balancr.cli.commands.logging.warning"
        ) as mock_warning:

            # Call function without overwrite
            result = commands._register_classifier_from_file(
                mock_registry, classifier_file, None, None, False
            )

            # Verify warning was logged about existing classifier
            mock_warning.assert_called_once()
            warning_msg = mock_warning.call_args[0][0]
            assert "already exists" in warning_msg
            assert "Use --overwrite" in warning_msg

            # Verify classifier was not registered
            mock_registry.register_custom_classifier.assert_not_called()

            # Verify result is empty
            assert result == []

    def test_register_from_file_with_registration_exception(self, tmp_path):
        """Test handling of exception during classifier registration."""
        # Create a temporary directory structure
        custom_dir = tmp_path / ".balancr" / "custom_classifiers"
        custom_dir.mkdir(parents=True, exist_ok=True)

        # Create a real Python file with a classifier class
        classifier_file = tmp_path / "error_classifier.py"
        code = """from sklearn.base import BaseEstimator

class ErrorClassifier(BaseEstimator):
    def __init__(self, param1=42):
        self.param1 = param1

    def fit(self, X, y):
        return self

    def predict(self, X):
        return [0] * len(X)
    """

        with open(classifier_file, "w") as f:
            f.write(code)

        # Create a mock registry that raises an exception on registration
        mock_registry = MagicMock()
        mock_registry.list_available_classifiers.return_value = {
            "custom": {"general": []},
            "sklearn": {},
        }
        mock_registry.register_custom_classifier.side_effect = Exception(
            "Registration error"
        )

        # Mock Path.home to return our temporary directory
        with patch("pathlib.Path.home", return_value=tmp_path), patch(
            "balancr.cli.commands.logging.error"
        ) as mock_error:

            # Call function
            result = commands._register_classifier_from_file(
                mock_registry, classifier_file, None, None, False
            )

            # Verify error was logged
            error_found = False
            for call in mock_error.call_args_list:
                if (
                    "Error registering classifier" in call[0][0]
                    and "Registration error" in call[0][0]
                ):
                    error_found = True
                    break
            assert error_found, "Error about registration failure wasn't logged"

            # Verify no classifiers were registered
            assert result == []

    def test_register_from_file_successful_registration(self, tmp_path):
        """Test successful registration of a classifier with metadata creation."""
        # Create a temporary directory structure
        custom_dir = tmp_path / ".balancr" / "custom_classifiers"
        custom_dir.mkdir(parents=True, exist_ok=True)

        # Create a real Python file with a classifier class
        classifier_file = tmp_path / "success_classifier.py"
        code = """from sklearn.base import BaseEstimator

class SuccessClassifier(BaseEstimator):
    def __init__(self, param1=42):
        self.param1 = param1

    def fit(self, X, y):
        return self

    def predict(self, X):
        return [0] * len(X)
    """

        with open(classifier_file, "w") as f:
            f.write(code)

        # Create a mock registry
        mock_registry = MagicMock()
        mock_registry.list_available_classifiers.return_value = {
            "custom": {"general": []},
            "sklearn": {},
        }

        # Create an empty metadata file
        metadata_file = custom_dir / "classifiers_metadata.json"
        with open(metadata_file, "w") as f:
            f.write("{}")

        # Mock Path.home to return our temporary directory
        with patch("pathlib.Path.home", return_value=tmp_path), patch(
            "balancr.cli.commands.datetime"
        ) as mock_datetime, patch(
            "balancr.cli.commands.logging.debug"
        ) as mock_debug:

            # Set fixed datetime for reproducibility
            mock_datetime.now.return_value.isoformat.return_value = (
                "2023-01-01T00:00:00"
            )

            # Call function
            result = commands._register_classifier_from_file(
                mock_registry, classifier_file, None, None, False
            )

            # Verify classifier was registered
            mock_registry.register_custom_classifier.assert_called_once()

            # Verify file was copied (check for debug log)
            file_copied = False
            for call in mock_debug.call_args_list:
                if "Copied" in call[0][0]:
                    file_copied = True
                    break
            assert file_copied, "Debug message about file copying wasn't logged"

            # Verify result contains the registered classifier
            assert "SuccessClassifier" in result

            # Verify metadata file was updated
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            assert "SuccessClassifier" in metadata
            assert "file" in metadata["SuccessClassifier"]
            assert "class_name" in metadata["SuccessClassifier"]
            assert metadata["SuccessClassifier"]["class_name"] == "SuccessClassifier"
            assert "registered_at" in metadata["SuccessClassifier"]
            assert (
                metadata["SuccessClassifier"]["registered_at"] == "2023-01-01T00:00:00"
            )

    @patch(
        "importlib.util.spec_from_file_location",
        side_effect=Exception("Unexpected failure"),
    )
    @patch("balancr.cli.commands.logging.error")
    def test_register_from_file_general_exception(self, mock_error_log, mock_spec):
        """Test generic exception handling during classifier loading."""
        mock_registry = MagicMock()
        mock_file_path = Path("/path/to/classifier_file.py")

        result = commands._register_classifier_from_file(
            mock_registry, mock_file_path, None, None, False
        )

        # Assert the general error was logged with the message and file path
        mock_error_log.assert_called_once()
        error_msg = mock_error_log.call_args[0][0]
        assert "Error processing file" in error_msg
        assert "Unexpected failure" in error_msg
        assert result == []


class TestRemoveClassifiersCommand:
    """Tests for the _remove_classifiers command."""

    @pytest.fixture
    def mock_custom_dir(self):
        """Patch Path.home() to point to a fake .balancr directory."""
        with patch("balancr.cli.commands.Path.home") as mock_home:
            mock_path = Path("/fake/home")
            mock_home.return_value = mock_path
            yield mock_path / ".balancr" / "custom_classifiers"

    @patch("balancr.cli.commands.logging.error")
    def test_no_metadata_file(self, mock_log, mock_custom_dir):
        """Test when metadata file does not exist."""
        args = MagicMock(remove_all=True)
        metadata_file = mock_custom_dir / "classifiers_metadata.json"
        metadata_file

        with patch("balancr.cli.commands.Path.exists", return_value=False):
            result = commands._remove_classifiers(args)

        mock_log.assert_called_once_with("No custom classifiers have been registered.")
        assert result == 1

    @patch("balancr.cli.commands.logging.error")
    def test_empty_metadata(self, mock_log, mock_custom_dir):
        """Test when metadata file exists but is empty."""
        args = MagicMock(remove_all=True)
        metadata_file = mock_custom_dir / "classifiers_metadata.json"
        metadata_file

        with patch(
            "balancr.cli.commands.Path.exists", return_value=True
        ), patch("builtins.open", mock_open(read_data="{}")):
            result = commands._remove_classifiers(args)

        mock_log.assert_called_once_with("No custom classifiers have been registered.")
        assert result == 1

    @patch("balancr.cli.commands.logging.warning")
    @patch("balancr.cli.commands.logging.info")
    def test_remove_all_classifiers(self, mock_info, mock_warning, mock_custom_dir):
        """Test removing all classifiers and deleting their files."""
        args = MagicMock(remove_all=True)
        metadata = {
            "clfA": {"file": "/path/to/a.py"},
            "clfB": {"file": "/path/to/b.py"},
            "clfC": {"file": "/path/to/a.py"},  # Same file as clfA
        }

        with patch(
            "balancr.cli.commands.Path.exists", return_value=True
        ), patch(
            "builtins.open", mock_open(read_data=json.dumps(metadata))
        ) as mock_file, patch(
            "balancr.cli.commands.Path.unlink"
        ) as mock_unlink, patch(
            "json.dump"
        ) as mock_dump:
            result = commands._remove_classifiers(args)
        mock_file

        mock_info.assert_called_once_with("Removing all custom classifiers...")
        # Should try to remove only two unique files
        mock_unlink.assert_any_call(missing_ok=True)
        assert mock_unlink.call_count == 2
        mock_dump.assert_called_once()
        assert result == 0

    @patch("balancr.cli.commands.logging.warning")
    @patch("balancr.cli.commands.Path")
    @patch("balancr.cli.commands.Path.exists", return_value=True)
    def test_remove_all_classifiers_file_removal_exception(
        self, mock_exists, mock_path_class, mock_logging_warning, mock_custom_dir
    ):
        """Test error handling when removing files in --remove-all path."""
        args = MagicMock(remove_all=True)
        mock_metadata = {
            "clfA": {"file": "/path/to/clfA.py"},
            "clfB": {"file": "/path/to/clfB.py"},
        }

        # Patch Path().unlink to raise an exception
        mock_path_instance = MagicMock()
        mock_path_instance.unlink.side_effect = Exception("Permission denied")
        mock_path_class.return_value = mock_path_instance

        with patch(
            "builtins.open", mock_open(read_data=json.dumps(mock_metadata))
        ), patch("json.dump"):
            result = commands._remove_classifiers(args)

        # Ensure warning was logged for the file removal failure
        mock_logging_warning.assert_any_call(
            "Error removing file /path/to/clfA.py: Permission denied"
        )
        assert result == 0

    @patch("balancr.cli.commands.logging.warning")
    def test_remove_specific_classifiers(self, mock_warning, mock_custom_dir):
        """Test removing specific classifiers and cleaning up unused files."""
        args = MagicMock(remove=["clfA", "clfX"], remove_all=False)
        metadata = {
            "clfA": {"file": "/path/to/a.py"},
            "clfB": {"file": "/path/to/b.py"},
        }

        # clfA should be removed, clfX should trigger a warning
        with patch(
            "balancr.cli.commands.Path.exists", return_value=True
        ), patch(
            "builtins.open", mock_open(read_data=json.dumps(metadata))
        ) as mock_file, patch(
            "balancr.cli.commands.Path.unlink"
        ) as mock_unlink, patch(
            "json.dump"
        ) as mock_dump:
            result = commands._remove_classifiers(args)
        mock_file
        mock_dump

        # Verify one warning for unknown classifier
        mock_warning.assert_any_call("Classifier 'clfX' not found.")
        # Verify one file was removed
        mock_unlink.assert_called_once_with(missing_ok=True)
        assert result == 0

    @patch("balancr.cli.commands.logging.error")
    def test_remove_specific_classifiers_none_found(self, mock_log, mock_custom_dir):
        """Test when no matching classifiers are found."""
        args = MagicMock(remove=["clfX"], remove_all=False)
        metadata = {
            "clfA": {"file": "/path/to/a.py"},
            "clfB": {"file": "/path/to/b.py"},
        }

        with patch(
            "balancr.cli.commands.Path.exists", return_value=True
        ), patch("builtins.open", mock_open(read_data=json.dumps(metadata))), patch(
            "balancr.cli.commands.Path.unlink"
        ) as mock_unlink, patch(
            "json.dump"
        ) as mock_dump:
            result = commands._remove_classifiers(args)
        mock_dump

        mock_log.assert_called_once_with("No matching classifiers were found.")
        mock_unlink.assert_not_called()
        assert result == 1

    @patch("balancr.cli.commands.logging.warning")
    @patch("balancr.cli.commands.Path")
    @patch("balancr.cli.commands.Path.exists", return_value=True)
    def test_remove_specific_classifier_file_removal_exception(
        self, mock_exists, mock_path_class, mock_logging_warning, mock_custom_dir
    ):
        """Test error handling when unlinking a file for a specific classifier."""
        args = MagicMock(remove=["clfA"], remove_all=False)
        mock_metadata = {"clfA": {"file": "/path/to/clfA.py"}}

        # Path().unlink will be triggered when removing clfA
        mock_path_instance = MagicMock()
        mock_path_instance.unlink.side_effect = Exception("File is in use")
        mock_path_class.return_value = mock_path_instance

        with patch(
            "builtins.open", mock_open(read_data=json.dumps(mock_metadata))
        ), patch("json.dump"):
            result = commands._remove_classifiers(args)

        # Should log the exception
        mock_logging_warning.assert_called_with(
            "Error removing file /path/to/clfA.py: File is in use"
        )
        assert result == 0
