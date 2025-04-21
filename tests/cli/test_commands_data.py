"""Tests for data-related commands in the balancr CLI."""

import pytest
from unittest.mock import patch, MagicMock

from balancr.cli import commands


@pytest.fixture
def mock_config_path(tmp_path):
    """Create a temporary config file path for testing."""
    return tmp_path / "test_config.json"


@pytest.fixture
def mock_framework():
    """Mock the BalancingFramework class."""
    mock = MagicMock()
    # Set up common methods
    mock.inspect_class_distribution.return_value = {0: 75, 1: 25}
    return mock


@pytest.fixture
def mock_data_file(tmp_path):
    """Create a temporary CSV data file for testing."""
    data_file = tmp_path / "test_data.csv"
    with open(data_file, "w") as f:
        f.write("feature1,feature2,target\n")
        f.write("1,2,0\n")
        f.write("3,4,1\n")
    return data_file


@pytest.fixture
def args_load_data(mock_config_path, mock_data_file):
    """Create mock arguments for load_data command."""
    args = MagicMock()
    args.file_path = str(mock_data_file)
    args.target_column = "target"
    args.feature_columns = ["feature1", "feature2"]
    args.config_path = str(mock_config_path)
    args.verbose = False
    return args


@pytest.fixture
def args_preprocess(mock_config_path):
    """Create mock arguments for preprocess command."""
    args = MagicMock()
    args.handle_missing = "mean"
    args.scale = "standard"
    args.encode = "auto"
    args.config_path = str(mock_config_path)
    return args


class TestLoadDataCommand:
    """Tests for the load_data command."""

    @patch("balancr.cli.commands.BalancingFramework")
    @patch("balancr.cli.config.update_config")
    def test_load_data_success(
        self, mock_update_config, mock_framework_class, args_load_data, mock_framework
    ):
        """Test successful data loading."""
        mock_framework_class.return_value = mock_framework

        result = commands.load_data(args_load_data)

        # Verify framework was initialised and methods called
        mock_framework_class.assert_called_once()
        mock_framework.load_data.assert_called_once_with(
            args_load_data.file_path,
            args_load_data.target_column,
            args_load_data.feature_columns,
            correlation_threshold=0.95,
        )
        mock_framework.inspect_class_distribution.assert_called_once()

        # Verify config was updated
        mock_update_config.assert_called_once()
        settings = mock_update_config.call_args[0][1]
        assert settings["data_file"] == args_load_data.file_path
        assert settings["target_column"] == args_load_data.target_column
        assert settings["feature_columns"] == args_load_data.feature_columns

        # Verify result
        assert result == 0

    @patch("balancr.cli.commands.logging.error")
    def test_load_data_file_not_found(self, mock_error, args_load_data):
        """Test handling of non-existent data file."""
        # Modify args to point to non-existent file
        args_load_data.file_path = "nonexistent_file.csv"

        # Call function
        result = commands.load_data(args_load_data)

        # Verify error was logged and correct result returned
        mock_error.assert_called_once()
        assert "File not found" in mock_error.call_args[0][0]
        assert result == 1

    @patch("balancr.cli.commands.BalancingFramework")
    @patch("balancr.cli.commands.logging.error")
    def test_load_data_framework_error(
        self, mock_error, mock_framework_class, args_load_data, mock_framework
    ):
        """Test handling of framework error during data loading."""
        # Set up mock to raise an exception
        mock_framework_class.return_value = mock_framework
        mock_framework.load_data.side_effect = Exception("Test error")

        # Call function
        result = commands.load_data(args_load_data)

        # Verify error was logged and correct result returned
        mock_error.assert_called_once()
        assert "Failed to load data" in mock_error.call_args[0][0]
        assert result == 1

    @patch("balancr.cli.commands.BalancingFramework", None)
    @patch("balancr.cli.commands.logging.error")
    def test_load_data_no_framework(self, mock_error, args_load_data):
        """Test handling when BalancingFramework is not available."""
        # Call function
        result = commands.load_data(args_load_data)

        # Config should still be updated even without framework
        assert result == 0


class TestPreprocessCommand:
    """Tests for the preprocess command."""

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.config.update_config")
    def test_preprocess_success(
        self, mock_update_config, mock_load_config, args_preprocess
    ):
        """Test successful preprocessing configuration."""
        # Set up load_config to return a valid config with data_file
        mock_load_config.return_value = {"data_file": "some_file.csv"}

        # Call the function
        result = commands.preprocess(args_preprocess)

        # Verify config was updated with correct settings
        mock_update_config.assert_called_once()
        settings = mock_update_config.call_args[0][1]
        assert (
            settings["preprocessing"]["handle_missing"]
            == args_preprocess.handle_missing
        )
        assert settings["preprocessing"]["scale"] == args_preprocess.scale
        assert settings["preprocessing"]["encode"] == args_preprocess.encode

        # Verify result
        assert result == 0

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.logging.error")
    def test_preprocess_no_data_file(
        self, mock_error, mock_load_config, args_preprocess
    ):
        """Test preprocessing without a configured data file."""
        # Set up load_config to return a config without data_file
        mock_load_config.return_value = {}

        # Call function
        result = commands.preprocess(args_preprocess)

        # Verify error was logged and function returned error code
        mock_error.assert_called_once()
        assert "Failed to configure preprocessing" in mock_error.call_args[0][0]
        assert result == 1

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.config.update_config")
    @patch("balancr.cli.commands.logging.error")
    def test_preprocess_config_error(
        self, mock_error, mock_update_config, mock_load_config, args_preprocess
    ):
        """Test handling of configuration errors during preprocessing."""
        # Set up load_config to return a valid config
        mock_load_config.return_value = {"data_file": "some_file.csv"}

        # Make update_config raise an exception
        mock_update_config.side_effect = Exception("Config error")

        # Call function
        result = commands.preprocess(args_preprocess)

        # Verify error was logged and correct result returned
        assert "Error analysing dataset for encoding"
        assert (
            "Failed to configure preprocessing: Config error"
            in mock_error.call_args[0][0]
        )
        assert result == 1

    @patch("balancr.cli.config.load_config")
    @patch("builtins.print")
    @patch("balancr.cli.config.update_config")
    def test_preprocess_output(
        self, mock_update_config, mock_print, mock_load_config, args_preprocess
    ):
        """Test that preprocessing configuration outputs correct information."""
        # Set up load_config to return a valid config
        mock_load_config.return_value = {"data_file": "some_file.csv"}

        # Call function
        commands.preprocess(args_preprocess)

        # Verify the output contains the configuration details
        assert mock_print.call_count >= 3  # At least header and 3 settings

        # Verify the settings were printed (any order)
        settings_output = " ".join(
            str(call[0][0]) for call in mock_print.call_args_list
        )
        assert args_preprocess.handle_missing in settings_output
        assert args_preprocess.scale in settings_output
        assert args_preprocess.encode in settings_output
