"""Tests for configuration-related commands in the balancr CLI."""

import pytest
from unittest.mock import patch, MagicMock

from balancr.cli import commands


@pytest.fixture
def mock_config_path(tmp_path):
    """Create a temporary config file path for testing."""
    return tmp_path / "test_config.json"


@pytest.fixture
def args_configure_metrics(mock_config_path):
    """Create mock arguments for configure_metrics command."""
    args = MagicMock()
    args.metrics = ["precision", "recall", "f1", "roc_auc"]
    args.save_formats = ["csv"]
    args.config_path = str(mock_config_path)
    args.verbose = False
    return args


class TestConfigureMetricsCommand:
    """Tests for the configure_metrics command."""

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.config.update_config")
    def test_configure_metrics_success(
        self, mock_update_config, mock_load_config, args_configure_metrics
    ):
        """Test successful metrics configuration."""
        # Set up mock to return an empty config
        mock_load_config.return_value = {}

        # Call function
        result = commands.configure_metrics(args_configure_metrics)

        # Verify config was updated with correct settings
        mock_update_config.assert_called_once()
        settings = mock_update_config.call_args[0][1]

        assert "output" in settings
        assert "metrics" in settings["output"]
        assert "save_metrics_formats" in settings["output"]
        assert settings["output"]["metrics"] == args_configure_metrics.metrics
        assert (
            settings["output"]["save_metrics_formats"]
            == args_configure_metrics.save_formats
        )

        # Verify function returned success
        assert result == 0

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.config.update_config")
    def test_configure_metrics_with_existing_output(
        self, mock_update_config, mock_load_config, args_configure_metrics
    ):
        """Test metrics configuration with existing output settings."""
        # Set up mock to return a config with existing output settings
        mock_load_config.return_value = {
            "output": {
                "visualisations": ["all"],
                "display_visualisations": True,
                "save_vis_formats": ["png"],
            }
        }

        # Call function
        result = commands.configure_metrics(args_configure_metrics)

        # Verify config was updated with correct settings
        mock_update_config.assert_called_once()
        settings = mock_update_config.call_args[0][1]

        assert "output" in settings
        assert "metrics" in settings["output"]
        assert "save_metrics_formats" in settings["output"]

        # Existing settings should be preserved
        assert "visualisations" in settings["output"]
        assert "display_visualisations" in settings["output"]
        assert "save_vis_formats" in settings["output"]

        # New settings should be present
        assert settings["output"]["metrics"] == args_configure_metrics.metrics
        assert (
            settings["output"]["save_metrics_formats"]
            == args_configure_metrics.save_formats
        )

        # Existing settings should be preserved
        assert settings["output"]["visualisations"] == ["all"]
        assert settings["output"]["display_visualisations"] is True
        assert settings["output"]["save_vis_formats"] == ["png"]

        # Verify function returned success
        assert result == 0

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.config.update_config")
    def test_configure_metrics_custom_values(
        self, mock_update_config, mock_load_config, args_configure_metrics
    ):
        """Test metrics configuration with custom metric values."""
        # Set custom metrics and save formats
        args_configure_metrics.metrics = ["precision", "g_mean", "specificity"]
        args_configure_metrics.save_formats = ["json", "csv"]

        # Set up mock to return an empty config
        mock_load_config.return_value = {}

        # Call function
        result = commands.configure_metrics(args_configure_metrics)

        # Verify config was updated with custom settings
        mock_update_config.assert_called_once()
        settings = mock_update_config.call_args[0][1]

        assert settings["output"]["metrics"] == ["precision", "g_mean", "specificity"]
        assert settings["output"]["save_metrics_formats"] == ["json", "csv"]

        # Verify function returned success
        assert result == 0

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.logging.error")
    def test_configure_metrics_load_config_error(
        self, mock_error, mock_load_config, args_configure_metrics
    ):
        """Test error handling when loading config fails."""
        # Make load_config raise an exception
        mock_load_config.side_effect = Exception("Config load error")

        # Call function
        result = commands.configure_metrics(args_configure_metrics)

        # Verify error was logged
        mock_error.assert_called_once()
        assert "Failed to configure metrics" in mock_error.call_args[0][0]

        # Verify function returned failure
        assert result == 1

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.config.update_config")
    @patch("balancr.cli.commands.logging.error")
    def test_configure_metrics_update_config_error(
        self, mock_error, mock_update_config, mock_load_config, args_configure_metrics
    ):
        """Test error handling when updating config fails."""
        # Set up mock to return an empty config
        mock_load_config.return_value = {}

        # Make update_config raise an exception
        mock_update_config.side_effect = Exception("Config update error")

        # Call function
        result = commands.configure_metrics(args_configure_metrics)

        # Verify error was logged
        mock_error.assert_called_once()
        assert "Failed to configure metrics" in mock_error.call_args[0][0]

        # Verify function returned failure
        assert result == 1


@pytest.fixture
def args_configure_visualisations(mock_config_path):
    """Create mock arguments for configure_visualisations command."""
    args = MagicMock()
    args.types = ["all"]
    args.display = False
    args.save_formats = ["png"]
    args.config_path = str(mock_config_path)
    args.verbose = False
    return args


class TestConfigureVisualisationsCommand:
    """Tests for the configure_visualisations command."""

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.config.update_config")
    def test_configure_visualisations_success(
        self, mock_update_config, mock_load_config, args_configure_visualisations
    ):
        """Test successful visualisations configuration."""
        # Set up mock to return an empty config
        mock_load_config.return_value = {}

        # Call the function
        result = commands.configure_visualisations(args_configure_visualisations)

        # Verify config was updated with correct settings
        mock_update_config.assert_called_once()
        settings = mock_update_config.call_args[0][1]

        assert "output" in settings
        assert "visualisations" in settings["output"]
        assert "display_visualisations" in settings["output"]
        assert "save_vis_formats" in settings["output"]
        assert (
            settings["output"]["visualisations"] == args_configure_visualisations.types
        )
        assert (
            settings["output"]["display_visualisations"]
            == args_configure_visualisations.display
        )
        assert (
            settings["output"]["save_vis_formats"]
            == args_configure_visualisations.save_formats
        )

        # Verify function returned success
        assert result == 0

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.config.update_config")
    def test_configure_visualisations_with_existing_output(
        self, mock_update_config, mock_load_config, args_configure_visualisations
    ):
        """Test visualisations configuration with existing output settings."""
        # Set up mock to return a config with existing output settings
        mock_load_config.return_value = {
            "output": {
                "metrics": ["precision", "recall", "f1"],
                "save_metrics_formats": ["csv", "json"],
            }
        }

        # Call the function
        result = commands.configure_visualisations(args_configure_visualisations)

        # Verify config was updated with correct settings
        mock_update_config.assert_called_once()
        settings = mock_update_config.call_args[0][1]

        assert "output" in settings

        # Existing settings should be preserved
        assert "metrics" in settings["output"]
        assert "save_metrics_formats" in settings["output"]

        # New settings should be present
        assert "visualisations" in settings["output"]
        assert "display_visualisations" in settings["output"]
        assert "save_vis_formats" in settings["output"]

        assert (
            settings["output"]["visualisations"] == args_configure_visualisations.types
        )
        assert (
            settings["output"]["display_visualisations"]
            == args_configure_visualisations.display
        )
        assert (
            settings["output"]["save_vis_formats"]
            == args_configure_visualisations.save_formats
        )

        # Existing settings should be preserved
        assert settings["output"]["metrics"] == ["precision", "recall", "f1"]
        assert settings["output"]["save_metrics_formats"] == ["csv", "json"]

        # Verify function returned success
        assert result == 0

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.config.update_config")
    def test_configure_visualisations_custom_values(
        self, mock_update_config, mock_load_config, args_configure_visualisations
    ):
        """Test visualisations configuration with custom values."""
        # Set custom visualisation types and save formats
        args_configure_visualisations.types = ["metrics", "distribution"]
        args_configure_visualisations.display = True
        args_configure_visualisations.save_formats = ["svg", "pdf"]

        # Set up mock to return an empty config
        mock_load_config.return_value = {}

        # Call the function
        result = commands.configure_visualisations(args_configure_visualisations)

        # Verify config was updated with custom settings
        mock_update_config.assert_called_once()
        settings = mock_update_config.call_args[0][1]

        assert settings["output"]["visualisations"] == ["metrics", "distribution"]
        assert settings["output"]["display_visualisations"] is True
        assert settings["output"]["save_vis_formats"] == ["svg", "pdf"]

        # Verify function returned success
        assert result == 0

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.logging.error")
    def test_configure_visualisations_load_config_error(
        self, mock_error, mock_load_config, args_configure_visualisations
    ):
        """Test error handling when loading config fails."""
        # Make load_config raise an exception
        mock_load_config.side_effect = Exception("Config load error")

        # Call the function
        result = commands.configure_visualisations(args_configure_visualisations)

        # Verify error was logged
        mock_error.assert_called_once()
        assert "Failed to configure visualisations" in mock_error.call_args[0][0]

        # Verify function returned failure
        assert result == 1

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.config.update_config")
    @patch("balancr.cli.commands.logging.error")
    def test_configure_visualisations_update_config_error(
        self,
        mock_error,
        mock_update_config,
        mock_load_config,
        args_configure_visualisations,
    ):
        """Test error handling when updating config fails."""
        # Set up mock to return an empty config
        mock_load_config.return_value = {}

        # Make update_config raise an exception
        mock_update_config.side_effect = Exception("Config update error")

        # Call the function
        result = commands.configure_visualisations(args_configure_visualisations)

        # Verify error was logged
        mock_error.assert_called_once()
        assert "Failed to configure visualisations" in mock_error.call_args[0][0]

        # Verify function returned failure
        assert result == 1


@pytest.fixture
def args_configure_evaluation(mock_config_path):
    """Create mock arguments for configure_evaluation command."""
    args = MagicMock()
    args.test_size = 0.2
    args.cross_validation = 0
    args.random_state = 42
    args.learning_curve_folds = 5
    args.learning_curve_points = 10
    args.config_path = str(mock_config_path)
    args.verbose = False
    return args


class TestConfigureEvaluationCommand:
    """Tests for the configure_evaluation command."""

    @patch("balancr.cli.config.update_config")
    def test_configure_evaluation_success(
        self, mock_update_config, args_configure_evaluation
    ):
        """Test successful evaluation configuration."""
        # Call the function
        result = commands.configure_evaluation(args_configure_evaluation)

        # Verify config was updated with correct settings
        mock_update_config.assert_called_once()
        settings = mock_update_config.call_args[0][1]

        assert "evaluation" in settings
        assert "test_size" in settings["evaluation"]
        assert "cross_validation" in settings["evaluation"]
        assert "random_state" in settings["evaluation"]
        assert "learning_curve_folds" in settings["evaluation"]
        assert "learning_curve_points" in settings["evaluation"]

        assert (
            settings["evaluation"]["test_size"] == args_configure_evaluation.test_size
        )
        assert (
            settings["evaluation"]["cross_validation"]
            == args_configure_evaluation.cross_validation
        )
        assert (
            settings["evaluation"]["random_state"]
            == args_configure_evaluation.random_state
        )
        assert (
            settings["evaluation"]["learning_curve_folds"]
            == args_configure_evaluation.learning_curve_folds
        )
        assert (
            settings["evaluation"]["learning_curve_points"]
            == args_configure_evaluation.learning_curve_points
        )

        # Verify function returned success
        assert result == 0

    @patch("balancr.cli.config.update_config")
    def test_configure_evaluation_custom_values(
        self, mock_update_config, args_configure_evaluation
    ):
        """Test evaluation configuration with custom values."""
        # Set custom evaluation parameters
        args_configure_evaluation.test_size = 0.3
        args_configure_evaluation.cross_validation = 5
        args_configure_evaluation.random_state = 123
        args_configure_evaluation.learning_curve_folds = 8
        args_configure_evaluation.learning_curve_points = 15

        # Call the function
        result = commands.configure_evaluation(args_configure_evaluation)

        # Verify config was updated with custom settings
        mock_update_config.assert_called_once()
        settings = mock_update_config.call_args[0][1]

        assert settings["evaluation"]["test_size"] == 0.3
        assert settings["evaluation"]["cross_validation"] == 5
        assert settings["evaluation"]["random_state"] == 123
        assert settings["evaluation"]["learning_curve_folds"] == 8
        assert settings["evaluation"]["learning_curve_points"] == 15

        # Verify function returned success
        assert result == 0

    @patch("balancr.cli.config.update_config")
    @patch("balancr.cli.commands.logging.error")
    def test_configure_evaluation_update_config_error(
        self, mock_error, mock_update_config, args_configure_evaluation
    ):
        """Test error handling when updating config fails."""
        # Make update_config raise an exception
        mock_update_config.side_effect = Exception("Config update error")

        # Call the function
        result = commands.configure_evaluation(args_configure_evaluation)

        # Verify error was logged
        mock_error.assert_called_once()
        assert "Failed to configure evaluation" in mock_error.call_args[0][0]

        # Verify function returned failure
        assert result == 1
