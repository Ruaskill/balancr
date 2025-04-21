"""Tests for technique-related commands in the balancr CLI."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from balancr.cli import commands
from balancr import BaseBalancer


class MockTechnique(BaseBalancer):
    """Mock technique class for testing."""

    def balance(self, X, y):
        return X, y


@pytest.fixture
def mock_config_path(tmp_path):
    """Create a temporary config file path for testing."""
    return tmp_path / "test_config.json"


@pytest.fixture
def mock_registry():
    """Mock the TechniqueRegistry class."""
    mock = MagicMock()
    mock.list_available_techniques.return_value = {
        "custom": ["CustomTechnique1", "CustomTechnique2"],
        "imblearn": ["SMOTE", "RandomUnderSampler", "ADASYN"],
    }
    mock.get_technique_default_params.return_value = {
        "sampling_strategy": "auto",
        "random_state": 42,
    }
    return mock


@pytest.fixture
def mock_technique_file(tmp_path):
    """Create a mock Python file with a technique class."""
    technique_file = tmp_path / "mock_technique.py"
    with open(technique_file, "w") as f:
        f.write(
            """
from balancr.base import BaseBalancer

class MockTechnique(BaseBalancer):
    def balance(self, X, y):
        return X, y
"""
        )
    return technique_file


@pytest.fixture
def args_select_techniques(mock_config_path):
    """Create mock arguments for select_techniques command."""
    args = MagicMock()
    args.techniques = ["SMOTE", "RandomUnderSampler"]
    args.list_available = False
    args.append = False
    args.config_path = str(mock_config_path)
    args.verbose = False
    return args


@pytest.fixture
def args_register_techniques(mock_config_path, mock_technique_file):
    """Create mock arguments for register_techniques command."""
    args = MagicMock()
    args.file_path = str(mock_technique_file)
    args.folder_path = None
    args.name = None
    args.class_name = None
    args.overwrite = False
    args.remove = None
    args.remove_all = False
    args.config_path = str(mock_config_path)
    args.verbose = False
    return args


class TestSelectTechniquesCommand:
    """Tests for the select_techniques command."""

    @patch("balancr.cli.commands.BalancingFramework")
    def test_list_available_techniques(
        self, mock_framework_class, args_select_techniques, mock_registry
    ):
        """Test listing available techniques."""
        # Set up for listing
        args_select_techniques.list_available = True
        args_select_techniques.techniques = []

        # Set up mock framework
        mock_framework = MagicMock()
        mock_framework.list_available_techniques.return_value = (
            mock_registry.list_available_techniques()
        )
        mock_framework_class.return_value = mock_framework

        # Mock print to capture output
        with patch("builtins.print") as mock_print:
            # Call function
            result = commands.select_techniques(args_select_techniques)

            # Verify framework was used to list techniques
            mock_framework_class.assert_called_once()
            mock_framework.list_available_techniques.assert_called_once()

            # Verify output includes both custom and imblearn techniques
            output = " ".join(str(call[0][0]) for call in mock_print.call_args_list)
            assert "Custom Techniques" in output
            assert "Imbalanced-Learn Techniques" in output
            assert "SMOTE" in output
            assert "RandomUnderSampler" in output

            # Verify result
            assert result == 0

    @patch("balancr.cli.commands.BalancingFramework")
    @patch("balancr.cli.config.load_config")
    def test_select_techniques_success(
        self,
        mock_load_config,
        mock_framework_class,
        args_select_techniques,
        mock_registry,
    ):
        """Test successful selection of techniques."""
        # Set up mock framework
        mock_framework = MagicMock()
        mock_framework.list_available_techniques.return_value = (
            mock_registry.list_available_techniques()
        )
        mock_framework.technique_registry = mock_registry
        mock_framework_class.return_value = mock_framework

        # Set up config loading
        mock_load_config.return_value = {}

        # Mock open and json.dump to capture the final configuration
        m_json_dump = MagicMock()

        with patch("builtins.open", mock_open()) as m_file, patch(
            "json.dump", m_json_dump
        ):

            # Call function
            result = commands.select_techniques(args_select_techniques)

            # Verify framework was used to validate techniques
            mock_framework_class.assert_called_once()
            mock_framework.list_available_techniques.assert_called_once()

            # Verify file was opened and json was written
            m_file.assert_called_once()
            m_json_dump.assert_called_once()

            # Verify the configuration that would be written
            config_arg = m_json_dump.call_args[0][0]
            assert "balancing_techniques" in config_arg
            assert "SMOTE" in config_arg["balancing_techniques"]
            assert "RandomUnderSampler" in config_arg["balancing_techniques"]

            # Verify result
            assert result == 0

    @patch("balancr.cli.commands.BalancingFramework")
    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.config.update_config")
    def test_select_techniques_append(
        self,
        mock_update_config,
        mock_load_config,
        mock_framework_class,
        args_select_techniques,
        mock_registry,
    ):
        """Test appending techniques to existing configuration."""
        # Set up for appending
        args_select_techniques.append = True

        # Set up mock framework
        mock_framework = MagicMock()
        mock_framework.list_available_techniques.return_value = (
            mock_registry.list_available_techniques()
        )
        mock_framework.technique_registry = mock_registry
        mock_framework_class.return_value = mock_framework

        # Set up existing config with some techniques
        mock_load_config.return_value = {
            "balancing_techniques": {"ADASYN": {"sampling_strategy": "auto"}}
        }

        # Call function
        result = commands.select_techniques(args_select_techniques)

        # Verify config was updated (not replaced)
        mock_update_config.assert_called_once()
        settings = mock_update_config.call_args[0][1]

        # Expected result should include both old and new techniques
        assert "balancing_techniques" in settings
        techniques = settings["balancing_techniques"]
        assert "ADASYN" in techniques  # Existing technique
        assert "SMOTE" in techniques  # New technique
        assert "RandomUnderSampler" in techniques  # New technique

        # Verify result
        assert result == 0

    @patch("balancr.cli.commands.BalancingFramework")
    @patch("balancr.cli.commands.logging.error")
    def test_select_invalid_techniques(
        self, mock_error, mock_framework_class, args_select_techniques
    ):
        """Test handling of invalid techniques selection."""
        # Set up mock framework with techniques
        mock_framework = MagicMock()
        mock_framework.list_available_techniques.return_value = {
            "custom": [],
            "imblearn": ["SMOTE"],  # Only SMOTE is available
        }
        mock_framework_class.return_value = mock_framework

        # Request an invalid technique
        args_select_techniques.techniques = ["InvalidTechnique"]

        # Call function
        result = commands.select_techniques(args_select_techniques)

        # Verify error was logged
        mock_error.assert_called_once()
        assert "Invalid techniques" in mock_error.call_args[0][0]

        # Verify result
        assert result == 1

    @patch("balancr.cli.commands.BalancingFramework", None)
    @patch("balancr.cli.config.load_config")
    def test_select_techniques_no_framework(
        self, mock_load_config, args_select_techniques
    ):
        """Test technique selection when framework is not available."""
        # Set up existing config
        mock_load_config.return_value = {}

        # Mock open and json.dump
        m_json_dump = MagicMock()

        with patch("builtins.open", mock_open()) as m_file, patch(
            "json.dump", m_json_dump
        ):
            m_file

            # Call function
            result = commands.select_techniques(args_select_techniques)

            # Verify result
            assert result == 0

            # Config should still be written, but without default parameters
            m_json_dump.assert_called_once()
            config_arg = m_json_dump.call_args[0][0]
            assert "balancing_techniques" in config_arg
            # When framework is not available, empty dict is used
            assert config_arg["balancing_techniques"] == {}

    @patch("balancr.cli.commands.BalancingFramework")
    def test_list_available_techniques_error(
        self, mock_framework_class, args_select_techniques
    ):
        """Test error handling when listing available techniques."""
        # Set up for listing
        args_select_techniques.list_available = True
        args_select_techniques.techniques = []

        # Make framework raise an exception
        mock_framework_class.return_value.list_available_techniques.side_effect = (
            Exception("Test error")
        )

        # Mock logging.error
        with patch("balancr.cli.commands.logging.error") as mock_error:
            # Call function
            result = commands.select_techniques(args_select_techniques)

            # Verify error was logged
            mock_error.assert_called_once()
            assert "Failed to list techniques" in mock_error.call_args[0][0]

            # Verify result
            assert result == 1

    @patch("balancr.cli.commands.BalancingFramework")
    @patch("balancr.cli.config.load_config")
    def test_select_techniques_error(
        self, mock_load_config, mock_framework_class, args_select_techniques
    ):
        """Test error handling in select_techniques."""
        # Configure the framework mock to avoid triggering validation errors
        mock_framework = MagicMock()
        mock_framework.list_available_techniques.return_value = {
            "custom": [],
            "imblearn": ["SMOTE", "RandomUnderSampler", "ADASYN"],
        }
        mock_framework_class.return_value = mock_framework

        # Make load_config return a value first but then raise an exception during update_config
        mock_load_config.return_value = {}

        with patch(
            "balancr.cli.config.update_config",
            side_effect=Exception("Test error"),
        ) as mock_update_config, patch(
            "balancr.cli.commands.logging.error"
        ) as mock_error:
            mock_update_config

            # Call function
            result = commands.select_techniques(args_select_techniques)

            # Verify error was logged
            mock_error.assert_called_once()
            assert "Failed to select techniques" in mock_error.call_args[0][0]

            # Verify result
            assert result == 1


class TestRegisterTechniquesCommand:
    """Tests for the register_techniques command."""

    @patch("balancr.cli.commands.TechniqueRegistry")
    @patch("balancr.cli.commands._register_from_file")
    def test_register_technique_from_file(
        self,
        mock_register_from_file,
        mock_registry_class,
        args_register_techniques,
        mock_registry,
    ):
        """Test registering a technique from a file."""
        # Set up mocks
        mock_registry_class.return_value = mock_registry
        mock_register_from_file.return_value = [
            "MockTechnique"
        ]  # Simulate successful registration

        # Mock Path.exists to return True for the file
        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_file", return_value=True
        ), patch("pathlib.Path.suffix", ".py", create=True):

            # Call function
            result = commands.register_techniques(args_register_techniques)

            # Verify _register_from_file was called
            mock_register_from_file.assert_called_once_with(
                mock_registry,
                Path(args_register_techniques.file_path),
                args_register_techniques.name,
                args_register_techniques.class_name,
                args_register_techniques.overwrite,
            )

            # Verify result
            assert result == 0

    @patch("balancr.cli.commands.TechniqueRegistry")
    @patch("balancr.cli.commands._register_from_file")
    def test_register_technique_with_custom_name(
        self,
        mock_register_from_file,
        mock_registry_class,
        args_register_techniques,
        mock_registry,
    ):
        """Test registering a technique with a custom name."""
        # Set up mocks
        mock_registry_class.return_value = mock_registry
        mock_register_from_file.return_value = [
            "MyCustomSMOTE"
        ]  # Simulate successful registration

        # Set custom name and class name
        args_register_techniques.name = "MyCustomSMOTE"
        args_register_techniques.class_name = "MockTechnique"

        # Mock Path.exists to return True for the file
        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_file", return_value=True
        ), patch("pathlib.Path.suffix", ".py", create=True):

            # Call function
            result = commands.register_techniques(args_register_techniques)

            # Verify _register_from_file was called with custom name and class
            mock_register_from_file.assert_called_once_with(
                mock_registry,
                Path(args_register_techniques.file_path),
                "MyCustomSMOTE",  # Custom name
                "MockTechnique",  # Class name
                args_register_techniques.overwrite,
            )

            # Verify result
            assert result == 0

    @patch("balancr.cli.commands.TechniqueRegistry")
    def test_register_technique_from_folder(
        self, mock_registry_class, args_register_techniques, mock_registry, tmp_path
    ):
        """Test registering techniques from a folder."""
        mock_registry_class.return_value = mock_registry

        # Create folder with Python files
        folder_path = tmp_path / "techniques"
        folder_path.mkdir()

        # Create a couple of Python files in the folder
        for i in range(2):
            with open(folder_path / f"technique{i}.py", "w") as f:
                f.write(
                    f"""
from balancr.base import BaseBalancer

class MockTechnique{i}(BaseBalancer):
    def balance(self, X, y):
        return X, y
"""
                )

        # Set up args for folder registration
        args_register_techniques.file_path = None
        args_register_techniques.folder_path = str(folder_path)

        # Mock Path.glob to return our Python files
        with patch.object(Path, "glob") as mock_glob:
            mock_glob.return_value = [
                folder_path / f"technique{i}.py" for i in range(2)
            ]

            # Mock _register_from_file to return success
            with patch(
                "balancr.cli.commands._register_from_file"
            ) as mock_register_from_file:
                mock_register_from_file.return_value = [
                    f"MockTechnique{i}" for i in range(2)
                ]

                # Mock path checks
                with patch("pathlib.Path.exists", return_value=True), patch(
                    "pathlib.Path.is_dir", return_value=True
                ):

                    # Call function
                    result = commands.register_techniques(args_register_techniques)

                    # Verify glob was called to find Python files
                    mock_glob.assert_called_once()

                    # Verify _register_from_file was called for each file
                    assert mock_register_from_file.call_count == 2

                    # Verify result
                    assert result == 0

    def test_register_techniques_remove_path(self, args_register_techniques):
        """Test register_techniques with remove option."""
        # Set up for removal
        args_register_techniques.file_path = None
        args_register_techniques.remove = ["CustomTechnique1"]

        # Mock _remove_techniques to return a known value
        with patch(
            "balancr.cli.commands._remove_techniques"
        ) as mock_remove:
            mock_remove.return_value = 42  # arbitrary return value

            # Call function
            result = commands.register_techniques(args_register_techniques)

            # Verify _remove_techniques was called and its result returned
            mock_remove.assert_called_once_with(args_register_techniques)
            assert result == 42

    def test_register_technique_not_python_file(self, args_register_techniques):
        """Test handling when file is not a Python file."""
        # Mock file path to exist but not be a Python file
        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_file", return_value=True
        ), patch("pathlib.Path.suffix", ".txt", create=True), patch(
            "balancr.cli.commands.logging.error"
        ) as mock_error:

            # Call function
            result = commands.register_techniques(args_register_techniques)

            # Verify error was logged
            mock_error.assert_called_once()
            assert "Not a Python file" in mock_error.call_args[0][0]

            # Verify result
            assert result == 1

    def test_register_technique_folder_not_found(self, args_register_techniques):
        """Test handling when folder path doesn't exist."""
        # Set up for folder registration with non-existent path
        args_register_techniques.file_path = None
        args_register_techniques.folder_path = "/nonexistent/folder"

        # Mock folder path to not exist
        with patch("pathlib.Path.exists", return_value=False), patch(
            "balancr.cli.commands.logging.error"
        ) as mock_error:

            # Call function
            result = commands.register_techniques(args_register_techniques)

            # Verify error was logged
            mock_error.assert_called_once()
            assert "Folder not found" in mock_error.call_args[0][0]

            # Verify result
            assert result == 1

    def test_register_technique_not_a_directory(self, args_register_techniques):
        """Test handling when path is not a directory."""
        # Set up for folder registration with path that's not a directory
        args_register_techniques.file_path = None
        args_register_techniques.folder_path = "/path/to/file.txt"

        # Mock path to exist but not be a directory
        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=False
        ), patch("balancr.cli.commands.logging.error") as mock_error:

            # Call function
            result = commands.register_techniques(args_register_techniques)

            # Verify error was logged
            mock_error.assert_called_once()
            assert "Not a directory" in mock_error.call_args[0][0]

            # Verify result
            assert result == 1

    def test_register_technique_no_valid_techniques_found(
        self, args_register_techniques, mock_registry
    ):
        """Test handling when no valid techniques are found."""
        # Set up for folder registration
        args_register_techniques.file_path = None
        args_register_techniques.folder_path = "/valid/folder"

        # Mock path checks and registry
        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=True
        ), patch("pathlib.Path.glob") as mock_glob, patch(
            "balancr.cli.commands.TechniqueRegistry"
        ) as mock_registry_class, patch(
            "balancr.cli.commands._register_from_file"
        ) as mock_register_from_file, patch(
            "balancr.cli.commands.logging.warning"
        ) as mock_warning:

            # Set up mock registry
            mock_registry_class.return_value = mock_registry

            # Set up mock glob to return one file
            mock_glob.return_value = [Path("/valid/folder/technique.py")]

            # _register_from_file returns empty list - no techniques found
            mock_register_from_file.return_value = []

            # Call function
            result = commands.register_techniques(args_register_techniques)

            # Verify warning was logged
            mock_warning.assert_called_once()
            assert "No valid balancing techniques found" in mock_warning.call_args[0][0]

            # Verify result
            assert result == 1

    def test_register_techniques_exception(self, args_register_techniques):
        """Test handling of exceptions in register_techniques."""
        # Make Path.exists raise an exception
        with patch("pathlib.Path.exists", side_effect=Exception("Test error")), patch(
            "balancr.cli.commands.logging.error"
        ) as mock_error, patch("traceback.print_exc") as mock_traceback:

            # Set up verbose mode
            args_register_techniques.verbose = True

            # Call function
            result = commands.register_techniques(args_register_techniques)

            # Verify error was logged
            mock_error.assert_called_once()
            assert "Error registering techniques" in mock_error.call_args[0][0]

            # Verify traceback was printed due to verbose mode
            mock_traceback.assert_called_once()

            # Verify result
            assert result == 1

    @patch("balancr.cli.commands.TechniqueRegistry")
    def test_register_technique_file_not_found(
        self, mock_registry_class, args_register_techniques, mock_registry
    ):
        """Test handling when technique file doesn't exist."""
        mock_registry_class.return_value = mock_registry

        # Set a non-existent file path
        args_register_techniques.file_path = "nonexistent_file.py"

        # Mock Path.exists to return False
        with patch("pathlib.Path.exists", return_value=False):

            # Call function
            result = commands.register_techniques(args_register_techniques)

            # Verify result
            assert result == 1

    @patch("balancr.cli.commands.TechniqueRegistry", None)
    def test_register_techniques_no_registry(self, args_register_techniques):
        """Test handling when TechniqueRegistry is not available."""
        # Call function
        result = commands.register_techniques(args_register_techniques)

        # Verify result
        assert result == 1


class TestRegisterFromFile:
    """Tests for the _register_from_file helper function."""

    @pytest.fixture
    def mock_module(self):
        """Create a mock module with technique classes."""
        mock = MagicMock()

        # Add two technique classes to the module
        class Technique1(BaseBalancer):
            def balance(self, X, y):
                return X, y

        class Technique2(BaseBalancer):
            def balance(self, X, y):
                return X, y

        # Set module name property
        mock.__name__ = "mock_module"

        # Add classes to the module mock
        mock.Technique1 = Technique1
        mock.Technique2 = Technique2

        # Configure inspect.getmembers to return our classes
        def mock_getmembers(module, predicate):
            return [("Technique1", Technique1), ("Technique2", Technique2)]

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
        """Test _register_from_file when no valid technique classes are found."""
        # Set up mocks
        mock_registry = MagicMock()
        mock_file_path = Path("/path/to/technique_file.py")

        # Create a module with no technique classes
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
        result = commands._register_from_file(
            mock_registry, mock_file_path, None, None, False
        )

        # Verify warning was logged
        mock_logging_warning.assert_called_once()
        assert (
            "No valid technique classes found" in mock_logging_warning.call_args[0][0]
        )

        # Verify no techniques were registered
        assert len(result) == 0

    def test_register_from_file_class_name_not_found_in_file(self, tmp_path):
        """Test when a specified class name is not found in the file."""
        # Create a temporary directory structure
        custom_dir = tmp_path / ".balancr" / "custom_techniques"
        custom_dir.mkdir(parents=True, exist_ok=True)

        # Create a real Python file with an actual technique class
        technique_file = tmp_path / "single_technique.py"
        code = """from balancr.base import BaseBalancer

class ExistingTechnique(BaseBalancer):
    def balance(self, X, y):
        return X, y
    """

        with open(technique_file, "w") as f:
            f.write(code)

        # Create a mock registry
        mock_registry = MagicMock()

        # Mock Path.home to return our temporary directory
        with patch("pathlib.Path.home", return_value=tmp_path):
            # Mock logging functions
            with patch("balancr.cli.commands.logging.error") as mock_error:
                # Call function with a non-existent class name
                result = commands._register_from_file(
                    mock_registry, technique_file, None, "NonExistentClass", False
                )

                # Verify the error was logged with the correct message
                mock_error.assert_called_once()
                error_msg = mock_error.call_args[0][0]
                assert "Class 'NonExistentClass' not found" in error_msg

                # Verify no techniques were registered
                assert result == []

    def test_register_from_file_custom_name_without_class_specified(self, tmp_path):
        """Test when a custom name is provided but multiple classes exist without specifying class_name."""
        # Create a temporary directory structure
        custom_dir = tmp_path / ".balancr" / "custom_techniques"
        custom_dir.mkdir(parents=True, exist_ok=True)

        # Create a real Python file with multiple technique classes - properly formatted
        technique_file = tmp_path / "multi_technique.py"
        code = """from balancr.base import BaseBalancer

class TechniqueOne(BaseBalancer):
    def balance(self, X, y):
        return X, y

class TechniqueTwo(BaseBalancer):
    def balance(self, X, y):
        return X, y
"""

        with open(technique_file, "w") as f:
            f.write(code)

        # Create a mock registry
        mock_registry = MagicMock()

        # Mock Path.home to return our temporary directory
        with patch("pathlib.Path.home", return_value=tmp_path):
            # Mock logging functions
            with patch("balancr.cli.commands.logging.error") as mock_error:
                # Call function with a custom name but no class specified
                result = commands._register_from_file(
                    mock_registry, technique_file, "CustomName", None, False
                )

                # Verify the error was logged
                mock_error.assert_called_once()
                error_msg = mock_error.call_args[0][0]
                assert "Multiple technique classes found" in error_msg
                assert "custom name provided" in error_msg
                assert "--class-name" in error_msg

                # Verify no techniques were registered
                assert result == []

    @patch("os.makedirs")
    @patch("importlib.util.spec_from_file_location")
    @patch("balancr.cli.commands.logging.error")
    def test_register_from_file_module_loading_error(
        self, mock_logging_error, mock_spec_from_file, mock_makedirs
    ):
        """Test _register_from_file when there's an error loading the module."""
        # Set up mocks
        mock_registry = MagicMock()
        mock_file_path = Path("/path/to/technique_file.py")

        # Make spec_from_file_location return None (error condition)
        mock_spec_from_file.return_value = None

        # Call function
        result = commands._register_from_file(
            mock_registry, mock_file_path, None, None, False
        )

        # Verify error was logged
        mock_logging_error.assert_called_once()
        assert "Could not load module" in mock_logging_error.call_args[0][0]

        # Verify no techniques were registered
        assert len(result) == 0

    @patch("balancr.cli.commands.TechniqueRegistry")
    @patch("balancr.cli.commands._register_from_file")
    def test_register_from_file_is_called_correctly(
        self, mock_register_from_file, mock_registry_class
    ):
        """Test that _register_from_file is called with the correct parameters."""
        # Setup
        mock_registry = MagicMock()
        mock_registry_class.return_value = mock_registry
        mock_register_from_file.return_value = [
            "MockTechnique"
        ]  # Mock successful registration

        # Create args for the test
        args = MagicMock()
        args.file_path = "/path/to/technique.py"
        args.name = "CustomName"
        args.class_name = "SomeClass"
        args.overwrite = True
        args.remove = None
        args.remove_all = False
        args.folder_path = None

        # Mock Path.exists and other path checks to return True
        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_file", return_value=True
        ), patch("pathlib.Path.suffix", ".py", create=True):

            # Call function
            result = commands.register_techniques(args)

            # Verify _register_from_file was called with the correct parameters
            mock_register_from_file.assert_called_once_with(
                mock_registry,
                Path("/path/to/technique.py"),
                "CustomName",
                "SomeClass",
                True,
            )

            # Verify successful result
            assert result == 0

    def test_register_from_file_with_custom_name_and_class_name(self, tmp_path):
        """Test registering a technique with a custom name for a specified class."""
        # Create a temporary directory structure
        custom_dir = tmp_path / ".balancr" / "custom_techniques"
        custom_dir.mkdir(parents=True, exist_ok=True)

        # Create a real Python file with multiple technique classes
        technique_file = tmp_path / "multi_technique.py"
        code = """from balancr.base import BaseBalancer

class TechniqueOne(BaseBalancer):
    def balance(self, X, y):
        return X, y

class TechniqueTwo(BaseBalancer):
    def balance(self, X, y):
        return X, y
    """

        with open(technique_file, "w") as f:
            f.write(code)

        # Create a mock registry
        mock_registry = MagicMock()
        mock_registry.list_available_techniques.return_value = {
            "custom": [],
            "imblearn": [],
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
            result = commands._register_from_file(
                mock_registry, technique_file, "CustomName", "TechniqueOne", False
            )

            # Verify technique was registered with custom name
            mock_registry.register_custom_technique.assert_called_once()
            args = mock_registry.register_custom_technique.call_args[0]
            assert args[0] == "CustomName"  # Custom name was used
            assert args[1].__name__ == "TechniqueOne"  # Correct class was selected

            # Verify success message was logged
            mock_info.assert_any_call("Successfully registered technique: CustomName")

            # Verify result contains the registered technique with custom name
            assert "CustomName" in result

    def test_register_from_file_technique_already_exists(self, tmp_path):
        """Test when technique already exists and overwrite is False."""
        # Create a temporary directory structure
        custom_dir = tmp_path / ".balancr" / "custom_techniques"
        custom_dir.mkdir(parents=True, exist_ok=True)

        # Create a real Python file with a technique class
        technique_file = tmp_path / "existing_technique.py"
        code = """from balancr.base import BaseBalancer

class ExistingTechnique(BaseBalancer):
    def balance(self, X, y):
        return X, y
    """

        with open(technique_file, "w") as f:
            f.write(code)

        # Create a mock registry that reports the technique already exists
        mock_registry = MagicMock()
        mock_registry.list_available_techniques.return_value = {
            "custom": ["ExistingTechnique"],
            "imblearn": [],
        }

        # Mock Path.home to return our temporary directory
        with patch("pathlib.Path.home", return_value=tmp_path), patch(
            "balancr.cli.commands.logging.warning"
        ) as mock_warning:

            # Call function without overwrite
            result = commands._register_from_file(
                mock_registry, technique_file, None, None, False
            )

            # Verify warning was logged about existing technique
            mock_warning.assert_called_once()
            warning_msg = mock_warning.call_args[0][0]
            assert "already exists" in warning_msg
            assert "Use --overwrite" in warning_msg

            # Verify technique was not registered
            mock_registry.register_custom_technique.assert_not_called()

            # Verify result is empty
            assert result == []

    def test_register_from_file_with_registration_exception(self, tmp_path):
        """Test handling of exception during technique registration."""
        # Create a temporary directory structure
        custom_dir = tmp_path / ".balancr" / "custom_techniques"
        custom_dir.mkdir(parents=True, exist_ok=True)

        # Create a real Python file with a technique class
        technique_file = tmp_path / "error_technique.py"
        code = """from balancr.base import BaseBalancer

class ErrorTechnique(BaseBalancer):
    def balance(self, X, y):
        return X, y
    """

        with open(technique_file, "w") as f:
            f.write(code)

        # Create a mock registry that raises an exception on registration
        mock_registry = MagicMock()
        mock_registry.list_available_techniques.return_value = {
            "custom": [],
            "imblearn": [],
        }
        mock_registry.register_custom_technique.side_effect = Exception(
            "Registration error"
        )

        # Mock Path.home to return our temporary directory
        with patch("pathlib.Path.home", return_value=tmp_path), patch(
            "balancr.cli.commands.logging.error"
        ) as mock_error:

            # Call function
            result = commands._register_from_file(
                mock_registry, technique_file, None, None, False
            )

            # Verify error was logged
            error_found = False
            for call in mock_error.call_args_list:
                if (
                    "Error registering technique" in call[0][0]
                    and "Registration error" in call[0][0]
                ):
                    error_found = True
                    break
            assert error_found, "Error about registration failure wasn't logged"

            # Verify no techniques were registered
            assert result == []

    def test_register_from_file_successful_registration(self, tmp_path):
        """Test successful registration of a technique with metadata creation."""
        # Create a temporary directory structure
        custom_dir = tmp_path / ".balancr" / "custom_techniques"
        custom_dir.mkdir(parents=True, exist_ok=True)

        # Create a real Python file with a technique class
        technique_file = tmp_path / "success_technique.py"
        code = """from balancr.base import BaseBalancer

class SuccessTechnique(BaseBalancer):
    def balance(self, X, y):
        return X, y
    """

        with open(technique_file, "w") as f:
            f.write(code)

        # Create a mock registry
        mock_registry = MagicMock()
        mock_registry.list_available_techniques.return_value = {
            "custom": [],
            "imblearn": [],
        }

        # Create an empty metadata file
        metadata_file = custom_dir / "techniques_metadata.json"
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
            result = commands._register_from_file(
                mock_registry, technique_file, None, None, False
            )

            # Verify technique was registered
            mock_registry.register_custom_technique.assert_called_once()

            # Verify file was copied (check for debug log)
            file_copied = False
            for call in mock_debug.call_args_list:
                if "Copied" in call[0][0]:
                    file_copied = True
                    break
            assert file_copied, "Debug message about file copying wasn't logged"

            # Verify result contains the registered technique
            assert "SuccessTechnique" in result

            # Verify metadata file was updated
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            assert "SuccessTechnique" in metadata
            assert "file" in metadata["SuccessTechnique"]
            assert "class_name" in metadata["SuccessTechnique"]
            assert metadata["SuccessTechnique"]["class_name"] == "SuccessTechnique"
            assert "registered_at" in metadata["SuccessTechnique"]
            assert (
                metadata["SuccessTechnique"]["registered_at"] == "2023-01-01T00:00:00"
            )


class TestRemoveTechniquesCommand:
    @patch("balancr.cli.commands.TechniqueRegistry")
    def test_remove_techniques(
        self, mock_registry_class, args_register_techniques, mock_registry
    ):
        """Test removing specific techniques."""
        mock_registry_class.return_value = mock_registry

        # Set up for removal
        args_register_techniques.file_path = None
        args_register_techniques.remove = ["CustomTechnique1"]

        # Mock metadata with existing techniques
        metadata = {
            "CustomTechnique1": {
                "file": "/path/to/technique1.py",
                "class_name": "Technique1",
            },
            "CustomTechnique2": {
                "file": "/path/to/technique2.py",
                "class_name": "Technique2",
            },
        }

        # Mock file operations
        with patch("pathlib.Path.exists") as mock_exists, patch(
            "pathlib.Path.unlink"
        ) as mock_unlink, patch("builtins.open", mock_open()), patch(
            "json.load", return_value=metadata
        ), patch(
            "json.dump"
        ) as mock_json_dump:

            # Set exists to True for metadata file
            mock_exists.return_value = True

            # Call function
            result = commands._remove_techniques(args_register_techniques)

            # Verify file was unlinked (deleted)
            mock_unlink.assert_called_once()

            # Verify result
            assert result == 0

            # Verify correct data was written back
            mock_json_dump.assert_called_once()
            new_metadata = mock_json_dump.call_args[0][0]
            assert "CustomTechnique1" not in new_metadata
            assert "CustomTechnique2" in new_metadata

    @patch("balancr.cli.commands.TechniqueRegistry")
    def test_remove_all_techniques(
        self, mock_registry_class, args_register_techniques, mock_registry
    ):
        """Test removing all techniques."""
        mock_registry_class.return_value = mock_registry

        # Set up for remove all
        args_register_techniques.file_path = None
        args_register_techniques.remove_all = True

        # Mock metadata with existing techniques
        metadata = {
            "CustomTechnique1": {
                "file": "/path/to/technique1.py",
                "class_name": "Technique1",
            },
            "CustomTechnique2": {
                "file": "/path/to/technique2.py",
                "class_name": "Technique2",
            },
        }

        # Mock file operations
        with patch("pathlib.Path.exists") as mock_exists, patch(
            "pathlib.Path.unlink"
        ) as mock_unlink, patch("builtins.open", mock_open()), patch(
            "json.load", return_value=metadata
        ), patch(
            "json.dump"
        ) as mock_json_dump:

            # Set exists to True for metadata file
            mock_exists.return_value = True

            # Call function
            result = commands._remove_techniques(args_register_techniques)

            # Verify files were unlinked (deleted)
            assert mock_unlink.call_count == 2

            # Verify result
            assert result == 0

            # Verify empty metadata was written back
            mock_json_dump.assert_called_once()
            new_metadata = mock_json_dump.call_args[0][0]
            assert new_metadata == {}

    @patch("balancr.cli.commands.logging")
    def test_metadata_file_missing(self, mock_logging):
        """Test when metadata file does not exist."""
        args = MagicMock()
        args.remove_all = False
        args.remove = ["CustomTechnique1"]

        with patch("pathlib.Path.exists", return_value=False):
            result = commands._remove_techniques(args)

        assert result == 1
        mock_logging.error.assert_called_once_with("No custom techniques have been registered.")

    @patch("balancr.cli.commands.logging")
    def test_empty_metadata_file(self, mock_logging):
        """Test when metadata file exists but is empty."""
        args = MagicMock()
        args.remove_all = False
        args.remove = ["CustomTechnique1"]

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open()
        ), patch("json.load", return_value={}):
            result = commands._remove_techniques(args)

        assert result == 1
        mock_logging.error.assert_called_once_with("No custom techniques have been registered.")

    @patch("balancr.cli.commands.logging")
    def test_exception_removing_file_remove_all(self, mock_logging):
        """Test file deletion exception when removing all techniques."""
        args = MagicMock()
        args.remove_all = True
        args.remove = []

        metadata = {
            "Tech1": {"file": "/fake/file1.py", "class_name": "SomeClass"},
        }

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open()
        ), patch("json.load", return_value=metadata), patch(
            "json.dump"
        ), patch(
            "pathlib.Path.unlink", side_effect=Exception("Delete failed")
        ):
            result = commands._remove_techniques(args)

        assert result == 0
        mock_logging.warning.assert_called_once()
        assert "Error removing file" in mock_logging.warning.call_args[0][0]

    @patch("balancr.cli.commands.logging")
    def test_exception_removing_file_specific_technique(self, mock_logging):
        """Test exception when removing a file for specific technique."""
        args = MagicMock()
        args.remove_all = False
        args.remove = ["Tech1"]

        metadata = {
            "Tech1": {"file": "/fake/file1.py", "class_name": "SomeClass"},
        }

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open()
        ), patch("json.load", return_value=metadata), patch(
            "json.dump"
        ), patch(
            "pathlib.Path.unlink", side_effect=Exception("Delete failed")
        ):
            result = commands._remove_techniques(args)

        assert result == 0
        mock_logging.warning.assert_called_once()
        assert "Error removing file" in mock_logging.warning.call_args[0][0]

    @patch("balancr.cli.commands.logging")
    def test_technique_not_found(self, mock_logging):
        """Test trying to remove a technique not in metadata."""
        args = MagicMock()
        args.remove_all = False
        args.remove = ["NonExistentTechnique"]

        metadata = {
            "Tech1": {"file": "/fake/file1.py", "class_name": "SomeClass"},
        }

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open()
        ), patch("json.load", return_value=metadata), patch(
            "json.dump"
        ):
            result = commands._remove_techniques(args)

        assert result == 1
        mock_logging.warning.assert_called_once_with("Technique 'NonExistentTechnique' not found.")

    @patch("balancr.cli.commands.logging")
    def test_no_matching_techniques_removed(self, mock_logging):
        """Test when no valid techniques are removed from metadata."""
        args = MagicMock()
        args.remove_all = False
        args.remove = ["NonExistent1", "NonExistent2"]

        metadata = {
            "ExistingTech": {"file": "/fake/file1.py", "class_name": "SomeClass"},
        }

        with patch("pathlib.Path.exists", return_value=True), patch(
            "builtins.open", mock_open()
        ), patch("json.load", return_value=metadata), patch(
            "json.dump"
        ):
            result = commands._remove_techniques(args)

        assert result == 1
        mock_logging.error.assert_called_once_with("No matching techniques were found.")
