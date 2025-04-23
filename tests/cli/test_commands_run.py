"""Tests for the run_comparison command in the balancr CLI."""

from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from balancr.cli import commands
from balancr import BalancingFramework


@pytest.fixture
def mock_config_path(tmp_path):
    """Create a temporary config file path for testing."""
    return tmp_path / "test_config.json"


@pytest.fixture
def mock_output_dir(tmp_path):
    """Create a temporary output directory for testing."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def minimal_config():
    """Create a minimal valid configuration."""
    return {
        "data_file": "test.csv",
        "target_column": "target",
        "balancing_techniques": {"SMOTE": {}, "RandomUnderSampler": {}},
    }


@pytest.fixture
def full_config():
    """Create a complete configuration with all options."""
    return {
        "data_file": "test.csv",
        "target_column": "target",
        "feature_columns": ["feature1", "feature2"],
        "preprocessing": {
            "handle_missing": "mean",
            "scale": "standard",
            "encode": "auto",
        },
        "balancing_techniques": {
            "SMOTE": {"k_neighbors": 5},
            "RandomUnderSampler": {"sampling_strategy": "auto"},
            "ADASYN": {"n_neighbors": 5},
        },
        "classifiers": {
            "RandomForestClassifier": {"n_estimators": 100, "random_state": 42},
            "LogisticRegression": {"C": 1.0, "random_state": 42},
        },
        "output": {
            "metrics": ["precision", "recall", "f1", "roc_auc"],
            "visualisations": ["all"],
            "display_visualisations": False,
            "save_metrics_formats": ["csv", "json"],
            "save_vis_formats": ["png", "pdf"],
        },
        "evaluation": {
            "test_size": 0.3,
            "cross_validation": 5,
            "random_state": 42,
            "learning_curve_folds": 5,
            "learning_curve_points": 10,
        },
    }


@pytest.fixture
def args_run_comparison(mock_config_path, mock_output_dir):
    """Create mock arguments for run_comparison command."""
    args = MagicMock()
    args.config_path = str(mock_config_path)
    args.output_dir = str(mock_output_dir)
    args.verbose = False
    return args


@pytest.fixture
def mock_framework():
    """Create a mock BalancingFramework instance."""
    mock = MagicMock(spec=BalancingFramework)

    # Configure common return values
    mock.inspect_class_distribution.return_value = {0: 80, 1: 20}
    mock.list_available_techniques.return_value = {
        "custom": [],
        "imblearn": ["SMOTE", "RandomUnderSampler", "ADASYN"],
    }

    # Set up results structure
    mock.train_classifiers.return_value = {
        "RandomForestClassifier": {
            "SMOTE": {
                "standard_metrics": {
                    "accuracy": 0.85,
                    "precision": 0.80,
                    "recall": 0.75,
                    "f1": 0.77,
                    "roc_auc": 0.82,
                }
            },
            "RandomUnderSampler": {
                "standard_metrics": {
                    "accuracy": 0.80,
                    "precision": 0.75,
                    "recall": 0.80,
                    "f1": 0.77,
                    "roc_auc": 0.78,
                }
            },
        }
    }

    return mock


class TestRunComparisonConfigChecks:
    """Tests for configuration validation and error handling in run_comparison."""

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.logging.error")
    def test_missing_configuration(
        self, mock_error, mock_load_config, args_run_comparison
    ):
        """Test handling of missing required configuration."""
        # Set up incomplete configuration
        mock_load_config.return_value = {
            "data_file": "test.csv"
        }  # Missing target_column and balancing_techniques

        # Call function
        result = commands.run_comparison(args_run_comparison)

        # Verify error was logged
        mock_error.assert_called_once()
        assert "Missing required configuration" in mock_error.call_args[0][0]

        # Verify result
        assert result == 1

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.logging.error")
    def test_missing_all_required_settings(
        self, mock_error, mock_load_config, args_run_comparison
    ):
        """Test handling when all required settings are missing."""
        # Set up empty configuration
        mock_load_config.return_value = {}

        # Call function
        result = commands.run_comparison(args_run_comparison)

        # Verify error was logged
        mock_error.assert_called_once()
        assert "Missing required configuration" in mock_error.call_args[0][0]
        assert "data_file" in mock_error.call_args[0][0]
        assert "target_column" in mock_error.call_args[0][0]
        assert "balancing_techniques" in mock_error.call_args[0][0]

        # Verify result
        assert result == 1

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.logging.error")
    def test_balancing_framework_not_available(
        self, mock_error, mock_load_config, args_run_comparison, minimal_config
    ):
        """Test handling when BalancingFramework is not available."""
        # Set up valid configuration
        mock_load_config.return_value = minimal_config

        # Make BalancingFramework unavailable
        with patch("balancr.cli.commands.BalancingFramework", None):
            # Call function
            result = commands.run_comparison(args_run_comparison)

            # Verify error was logged
            mock_error.assert_called_once()
            assert "Balancing framework not available" in mock_error.call_args[0][0]

            # Verify result
            assert result == 1

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.logging.error")
    def test_config_loading_exception(
        self, mock_error, mock_load_config, args_run_comparison
    ):
        """Test handling of exceptions during config loading."""
        # Make load_config raise an exception
        mock_load_config.side_effect = Exception("Config error")

        # Call function
        result = commands.run_comparison(args_run_comparison)

        # Verify error was logged
        mock_error.assert_called_once()
        assert "Failed to load configuration" in mock_error.call_args[0][0]
        assert "Config error" in mock_error.call_args[0][0]

        # Verify result
        assert result == 1


class TestRunComparisonDataLoading:
    """Tests for data loading and preprocessing in run_comparison."""

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.BalancingFramework")
    @patch("balancr.cli.commands.logging.info")
    def test_data_loading(
        self,
        mock_info,
        mock_balancing_framework,
        mock_load_config,
        args_run_comparison,
        minimal_config,
        mock_framework,
    ):
        """Test loading data with minimal configuration."""
        # Set up configuration
        mock_load_config.return_value = minimal_config

        # Set up mock framework instance
        mock_balancing_framework.return_value = mock_framework

        # Mock Path.mkdir to track creation of output directory
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            # Call function but interrupt after data loading
            with patch.object(
                mock_framework,
                "apply_balancing_techniques",
                side_effect=KeyboardInterrupt,
            ):
                try:
                    commands.run_comparison(args_run_comparison)
                except KeyboardInterrupt:
                    pass

            # Verify output directory was created
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

            # Verify data loading was called with correct parameters
            mock_framework.load_data.assert_called_once_with(
                minimal_config["data_file"],
                minimal_config["target_column"],
                None,  # No feature_columns in minimal_config
            )

            # Verify log message
            mock_info.assert_any_call(
                f"Loading data from {minimal_config['data_file']}"
            )

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.BalancingFramework")
    def test_data_loading_with_feature_columns(
        self,
        mock_balancing_framework,
        mock_load_config,
        args_run_comparison,
        full_config,
        mock_framework,
    ):
        """Test loading data with specified feature columns."""
        # Set up configuration
        mock_load_config.return_value = full_config

        # Set up mock framework instance
        mock_balancing_framework.return_value = mock_framework

        # Call function but interrupt after data loading
        with patch.object(
            mock_framework, "apply_balancing_techniques", side_effect=KeyboardInterrupt
        ):
            try:
                commands.run_comparison(args_run_comparison)
            except KeyboardInterrupt:
                pass

        # Verify data loading was called with feature columns
        mock_framework.load_data.assert_called_once_with(
            full_config["data_file"],
            full_config["target_column"],
            full_config["feature_columns"],
        )

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.BalancingFramework")
    @patch("balancr.cli.commands.logging.info")
    def test_preprocessing(
        self,
        mock_info,
        mock_balancing_framework,
        mock_load_config,
        args_run_comparison,
        full_config,
        mock_framework,
    ):
        """Test preprocessing with full configuration."""
        # Set up configuration
        mock_load_config.return_value = full_config

        # Set up mock framework instance
        mock_balancing_framework.return_value = mock_framework

        # Call function but interrupt after preprocessing
        with patch.object(
            mock_framework, "apply_balancing_techniques", side_effect=KeyboardInterrupt
        ):
            try:
                commands.run_comparison(args_run_comparison)
            except KeyboardInterrupt:
                pass

        # Verify preprocessing was called with correct parameters
        mock_framework.preprocess_data.assert_called_once_with(
            handle_missing=full_config["preprocessing"]["handle_missing"],
            scale=full_config["preprocessing"]["scale"],
            categorical_features={},
            hash_components_dict={},
            handle_constant_features=None,
            handle_correlations=None,
        )

        # Verify log message
        mock_info.assert_any_call("Applying preprocessing...")
        mock_info.assert_any_call("Data preprocessing applied")

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.BalancingFramework")
    @patch("balancr.cli.commands.logging.info")
    def test_no_preprocessing_config(
        self,
        mock_info,
        mock_balancing_framework,
        mock_load_config,
        args_run_comparison,
        minimal_config,
        mock_framework,
    ):
        """Test behaviour when preprocessing configuration is missing."""
        # Set up configuration without preprocessing section
        mock_load_config.return_value = minimal_config

        # Set up mock framework instance
        mock_balancing_framework.return_value = mock_framework

        # Call function but interrupt after data loading
        with patch.object(
            mock_framework, "apply_balancing_techniques", side_effect=KeyboardInterrupt
        ):
            try:
                commands.run_comparison(args_run_comparison)
            except KeyboardInterrupt:
                pass

        # Verify preprocessing was not called
        mock_framework.preprocess_data.assert_not_called()

        # Check that specific preprocessing log messages don't appear
        preprocessing_messages = [
            "Applying preprocessing...",
            "Data preprocessing applied",
        ]
        for message in preprocessing_messages:
            for call_args in mock_info.call_args_list:
                if call_args[0][0] == message:
                    pytest.fail(f"Unexpected preprocessing message: {message}")

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.BalancingFramework")
    @patch("balancr.cli.commands.logging.error")
    def test_data_loading_exception(
        self,
        mock_error,
        mock_balancing_framework,
        mock_load_config,
        args_run_comparison,
        minimal_config,
        mock_framework,
    ):
        """Test handling of exceptions during data loading."""
        # Set up configuration
        mock_load_config.return_value = minimal_config

        # Set up mock framework instance
        mock_balancing_framework.return_value = mock_framework

        # Make load_data raise an exception
        mock_framework.load_data.side_effect = Exception("File not found")

        # Call function
        result = commands.run_comparison(args_run_comparison)

        # Verify error was logged
        mock_error.assert_called_once()
        assert "Error during comparison" in mock_error.call_args[0][0]
        assert "File not found" in mock_error.call_args[0][0]

        # Verify result
        assert result == 1

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.BalancingFramework")
    @patch("balancr.cli.commands.logging.error")
    def test_data_loading_exception_verbose_mode(
        self,
        mock_error,
        mock_balancing_framework,
        mock_load_config,
        args_run_comparison,
        minimal_config,
        mock_framework,
    ):
        """Test exception handling with verbose mode enabled."""
        # Set up configuration
        mock_load_config.return_value = minimal_config

        # Enable verbose mode
        args_run_comparison.verbose = True

        # Set up mock framework instance
        mock_balancing_framework.return_value = mock_framework

        # Make load_data raise an exception
        mock_framework.load_data.side_effect = Exception("File not found")

        # Mock traceback.print_exc to verify it's called
        with patch("traceback.print_exc") as mock_traceback:
            # Call function
            result = commands.run_comparison(args_run_comparison)

            # Verify traceback was printed due to verbose mode
            mock_traceback.assert_called_once()

        # Verify error was logged
        mock_error.assert_called_once()
        assert "Error during comparison" in mock_error.call_args[0][0]
        assert "File not found" in mock_error.call_args[0][0]

        # Verify result
        assert result == 1


class TestRunComparisonBalancingTechniques:
    """Tests for applying balancing techniques in run_comparison."""

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.BalancingFramework")
    @patch("balancr.cli.commands.logging.info")
    def test_apply_balancing_techniques(
        self,
        mock_info,
        mock_balancing_framework,
        mock_load_config,
        args_run_comparison,
        minimal_config,
        mock_framework,
    ):
        """Test applying balancing techniques with minimal configuration."""
        # Set up configuration
        mock_load_config.return_value = minimal_config

        # Set up mock framework instance
        mock_balancing_framework.return_value = mock_framework

        # Call function but interrupt after applying techniques
        with patch.object(
            mock_framework, "train_classifiers", side_effect=KeyboardInterrupt
        ):
            try:
                commands.run_comparison(args_run_comparison)
            except KeyboardInterrupt:
                pass

        # Verify balancing techniques were applied with correct parameters
        mock_framework.apply_balancing_techniques.assert_called_once_with(
            list(minimal_config["balancing_techniques"].keys()),
            test_size=0.2,  # Default value
            random_state=42,  # Default value
            technique_params=minimal_config["balancing_techniques"],
            include_original=False,
        )

        # Verify log messages
        mock_info.assert_any_call(
            f"Running comparison with techniques: {', '.join(list(minimal_config['balancing_techniques'].keys()))}"
        )
        mock_info.assert_any_call("Applying balancing techniques...")

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.BalancingFramework")
    def test_apply_balancing_techniques_with_full_config(
        self,
        mock_balancing_framework,
        mock_load_config,
        args_run_comparison,
        full_config,
        mock_framework,
    ):
        """Test applying balancing techniques with full configuration."""
        # Set up configuration
        mock_load_config.return_value = full_config

        # Set up mock framework instance
        mock_balancing_framework.return_value = mock_framework

        # Call function but interrupt after applying techniques
        with patch.object(
            mock_framework, "train_classifiers", side_effect=KeyboardInterrupt
        ):
            try:
                commands.run_comparison(args_run_comparison)
            except KeyboardInterrupt:
                pass

        # Verify balancing techniques were applied with config values
        mock_framework.apply_balancing_techniques.assert_called_once_with(
            list(full_config["balancing_techniques"].keys()),
            test_size=full_config["evaluation"]["test_size"],
            random_state=full_config["evaluation"]["random_state"],
            technique_params=full_config["balancing_techniques"],
            include_original=False,
        )

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.BalancingFramework")
    @patch("balancr.cli.commands.logging.info")
    def test_generate_balanced_datasets(
        self,
        mock_info,
        mock_balancing_framework,
        mock_load_config,
        args_run_comparison,
        mock_output_dir,
        minimal_config,
        mock_framework,
    ):
        """Test generation of balanced datasets."""
        # Set up configuration
        mock_load_config.return_value = minimal_config

        # Set up mock framework instance
        mock_balancing_framework.return_value = mock_framework

        # Call function but interrupt after generating balanced datasets
        with patch.object(
            mock_framework, "train_classifiers", side_effect=KeyboardInterrupt
        ):
            try:
                commands.run_comparison(args_run_comparison)
            except KeyboardInterrupt:
                pass

        # Verify balanced datasets were generated
        balanced_dir = Path(args_run_comparison.output_dir) / "balanced_datasets"
        mock_framework.generate_balanced_data.assert_called_once_with(
            folder_path=str(balanced_dir),
            techniques=list(minimal_config["balancing_techniques"].keys()),
            file_format="csv",
        )

        # Verify log message
        mock_info.assert_any_call(f"Saving balanced datasets to {balanced_dir}")

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.BalancingFramework")
    @patch("balancr.cli.commands.logging.error")
    def test_apply_balancing_techniques_exception(
        self,
        mock_error,
        mock_balancing_framework,
        mock_load_config,
        args_run_comparison,
        minimal_config,
        mock_framework,
    ):
        """Test handling of exceptions when applying balancing techniques."""
        # Set up configuration
        mock_load_config.return_value = minimal_config

        # Set up mock framework instance
        mock_balancing_framework.return_value = mock_framework

        # Make apply_balancing_techniques raise an exception
        mock_framework.apply_balancing_techniques.side_effect = Exception(
            "Invalid technique"
        )

        # Call function
        result = commands.run_comparison(args_run_comparison)

        # Verify error was logged
        mock_error.assert_called_once()
        assert "Error during comparison" in mock_error.call_args[0][0]
        assert "Invalid technique" in mock_error.call_args[0][0]

        # Verify result
        assert result == 1


class TestRunComparisonClassifierTraining:
    """Tests for training classifiers and evaluating results in run_comparison."""

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.BalancingFramework")
    @patch("balancr.cli.commands.logging.info")
    def test_train_classifiers(
        self,
        mock_info,
        mock_balancing_framework,
        mock_load_config,
        args_run_comparison,
        minimal_config,
        mock_framework,
    ):
        """Test training classifiers with minimal configuration (default RandomForestClassifier)."""
        # Set up configuration
        mock_load_config.return_value = minimal_config

        # Set up mock framework instance
        mock_balancing_framework.return_value = mock_framework

        # Ensure train_classifiers completes without interruption
        mock_framework.train_classifiers.side_effect = None

        # Call function but interrupt after classifier training
        with patch.object(
            mock_framework, "save_classifier_results", side_effect=KeyboardInterrupt
        ):
            try:
                commands.run_comparison(args_run_comparison)
            except KeyboardInterrupt:
                pass

        # Verify train_classifiers was called
        mock_framework.train_classifiers.assert_called_once()

        # Extract call kwargs
        call_args = mock_framework.train_classifiers.call_args[1]

        # Verify that classifier_configs exists
        assert "classifier_configs" in call_args

        # Check that enable_cv is false by default
        assert call_args.get("enable_cv") is False

        # Verify log message
        mock_info.assert_any_call("Training classifiers on balanced datasets...")

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.BalancingFramework")
    def test_train_classifiers_with_full_config(
        self,
        mock_balancing_framework,
        mock_load_config,
        args_run_comparison,
        full_config,
        mock_framework,
    ):
        """Test training classifiers with custom classifiers and cross-validation."""
        # Set up configuration
        mock_load_config.return_value = full_config

        # Set up mock framework instance
        mock_balancing_framework.return_value = mock_framework

        # Call function but interrupt after classifier training
        with patch.object(
            mock_framework, "save_classifier_results", side_effect=KeyboardInterrupt
        ):
            try:
                commands.run_comparison(args_run_comparison)
            except KeyboardInterrupt:
                pass

        # Verify train_classifiers was called with configured parameters
        mock_framework.train_classifiers.assert_called_once()
        # Extract the call arguments
        args, kwargs = mock_framework.train_classifiers.call_args
        # Check for configured classifiers
        assert kwargs.get("classifier_configs") == full_config["classifiers"]
        # Check for cross-validation
        assert kwargs.get("enable_cv") is True
        assert kwargs.get("cv_folds") == full_config["evaluation"]["cross_validation"]

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.BalancingFramework")
    @patch("balancr.cli.commands.logging.warning")
    def test_train_classifiers_no_configured_classifiers(
        self,
        mock_warning,
        mock_balancing_framework,
        mock_load_config,
        args_run_comparison,
        minimal_config,
        mock_framework,
    ):
        """Test training classifiers with no configured classifiers (uses default)."""
        # Set up configuration
        mock_load_config.return_value = minimal_config

        # Set up mock framework instance
        mock_balancing_framework.return_value = mock_framework

        # Call function but interrupt after classifier training
        with patch.object(
            mock_framework, "save_classifier_results", side_effect=KeyboardInterrupt
        ):
            try:
                commands.run_comparison(args_run_comparison)
            except KeyboardInterrupt:
                pass

        # Verify warning was logged
        mock_warning.assert_any_call(
            "No classifiers configured. Using default RandomForestClassifier."
        )

        # Verify train_classifiers was called with default RandomForestClassifier
        mock_framework.train_classifiers.assert_called_once()

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.BalancingFramework")
    @patch("balancr.cli.commands.logging.error")
    def test_train_classifiers_exception(
        self,
        mock_error,
        mock_balancing_framework,
        mock_load_config,
        args_run_comparison,
        minimal_config,
        mock_framework,
    ):
        """Test handling of exceptions during classifier training."""
        # Set up configuration
        mock_load_config.return_value = minimal_config

        # Set up mock framework instance
        mock_balancing_framework.return_value = mock_framework

        # Make train_classifiers raise an exception
        mock_framework.train_classifiers.side_effect = Exception("Training error")

        # Call function
        result = commands.run_comparison(args_run_comparison)

        # Verify error was logged
        mock_error.assert_called_once()
        assert "Error during comparison" in mock_error.call_args[0][0]
        assert "Training error" in mock_error.call_args[0][0]

        # Verify result
        assert result == 1


class TestRunComparisonVisualisations:
    """Tests for visualisation generation in run_comparison."""

    @pytest.fixture
    def mock_visualisation_functions(self):
        """Patch all visualisation-related functions."""
        with patch(
            "balancr.cli.commands.plot_comparison_results"
        ) as mock_plot_comparison:
            yield mock_plot_comparison

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.BalancingFramework")
    def test_visualise_class_distributions(
        self,
        mock_balancing_framework,
        mock_load_config,
        args_run_comparison,
        full_config,
        mock_framework,
    ):
        """Test generation of class distribution visualisations."""
        # Set up configuration
        mock_load_config.return_value = full_config

        # Set up mock framework instance
        mock_balancing_framework.return_value = mock_framework

        # Reset the mocks to clear any previous calls
        mock_framework.inspect_class_distribution.reset_mock()
        mock_framework.compare_balanced_class_distributions.reset_mock()

        # We won't patch the actual framework methods we want to test
        with patch.object(mock_framework, "save_classifier_results"), patch.object(
            mock_framework, "generate_learning_curves"
        ), patch("balancr.cli.commands.plot_comparison_results"):

            # Run comparison
            commands.run_comparison(args_run_comparison)

        # Test that inspect_class_distribution was called
        assert mock_framework.inspect_class_distribution.called

        # Test that compare_balanced_class_distributions was called
        assert mock_framework.compare_balanced_class_distributions.called

        # Let's check the save_path arguments for correct format types
        for format_type in full_config["output"]["save_vis_formats"]:
            if format_type == "none":
                continue

            # For each call, check that the format type shows up in a save_path
            for call in mock_framework.inspect_class_distribution.call_args_list:
                kwargs = call[1]
                if "save_path" in kwargs and format_type in str(kwargs["save_path"]):
                    break
            else:
                # If we didn't break, the format wasn't found
                pytest.fail(
                    f"No save_path with format {format_type} for inspect_class_distribution"
                )

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.BalancingFramework")
    def test_visualise_metrics(
        self,
        mock_balancing_framework,
        mock_load_config,
        args_run_comparison,
        full_config,
        mock_framework,
        mock_visualisation_functions,
    ):
        """Test generation of metrics visualisations."""
        # Set up configuration
        mock_load_config.return_value = full_config

        # Set up mock framework instance
        mock_balancing_framework.return_value = mock_framework

        # Call function but force completion
        with patch.object(mock_framework, "save_classifier_results"), patch.object(
            mock_framework, "inspect_class_distribution"
        ), patch.object(
            mock_framework, "compare_balanced_class_distributions"
        ), patch.object(
            mock_framework, "generate_learning_curves"
        ), patch(
            "balancr.cli.commands.plot_radar_chart"  # Add mock for direct radar chart call
        ) as mock_radar_chart:
            # Run comparison with normal full completion expected
            commands.run_comparison(args_run_comparison)

        # Verify metrics visualisations were generated for each classifier and format
        for classifier_name in full_config["classifiers"].keys():
            for format_type in full_config["output"]["save_vis_formats"]:
                if format_type == "none":
                    continue

                # Look for either metrics_comparison OR radar_chart calls
                metrics_calls = [
                    call
                    for call in mock_visualisation_functions.call_args_list
                    if classifier_name in str(call)
                    and (
                        f"metrics_comparison.{format_type}" in str(call)
                        or f"radar_chart.{format_type}" in str(call)
                    )
                ]

                # If no metrics_comparison calls, check for radar_chart calls
                if not metrics_calls:
                    metrics_calls = [
                        call
                        for call in mock_radar_chart.call_args_list
                        if classifier_name in str(call)
                        and f".{format_type}" in str(call)
                    ]

                assert len(metrics_calls) > 0

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.BalancingFramework")
    def test_no_visualisations(
        self,
        mock_balancing_framework,
        mock_load_config,
        args_run_comparison,
        minimal_config,
        mock_framework,
    ):
        """Test when visualisations are disabled ('none')."""
        # Set up configuration with visualisations disabled
        config = dict(minimal_config)
        config["output"] = {"visualisations": ["none"], "save_vis_formats": ["png"]}
        mock_load_config.return_value = config

        # Set up mock framework instance
        mock_balancing_framework.return_value = mock_framework

        # Call function but force completion
        with patch.object(mock_framework, "save_classifier_results"), patch.object(
            mock_framework, "inspect_class_distribution"
        ), patch.object(
            mock_framework, "compare_balanced_class_distributions"
        ), patch.object(
            mock_framework, "generate_learning_curves"
        ), patch(
            "balancr.cli.commands.plot_comparison_results"
        ):
            # Run comparison with normal full completion expected
            commands.run_comparison(args_run_comparison)

        # Verify no visualisations were generated
        mock_framework.inspect_class_distribution.assert_not_called()
        mock_framework.compare_balanced_class_distributions.assert_not_called()
        mock_framework.generate_learning_curves.assert_not_called()

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.BalancingFramework")
    def test_visualisation_display_enabled(
        self,
        mock_balancing_framework,
        mock_load_config,
        args_run_comparison,
        full_config,
        mock_framework,
    ):
        """Test when visualisation display is enabled."""
        # Set up configuration with display enabled
        config = dict(full_config)
        config["output"]["display_visualisations"] = True
        mock_load_config.return_value = config

        # Set up mock framework instance
        mock_balancing_framework.return_value = mock_framework

        # Call function but force completion
        with patch.object(mock_framework, "save_classifier_results"), patch.object(
            mock_framework, "inspect_class_distribution"
        ), patch.object(
            mock_framework, "compare_balanced_class_distributions"
        ), patch.object(
            mock_framework, "generate_learning_curves"
        ), patch(
            "balancr.cli.commands.plot_comparison_results"
        ), patch(
            "balancr.cli.commands.plot_radar_chart"  # Add mock for direct radar chart call
        ) as mock_radar_chart:
            # Run comparison with normal full completion expected
            commands.run_comparison(args_run_comparison)

        # Verify display=True was passed to visualisation functions
        for call_args in mock_framework.inspect_class_distribution.call_args_list:
            kwargs = call_args[1]
            assert kwargs.get("display") is True

        for call_args in mock_radar_chart:
            kwargs = call_args[1]
            assert kwargs.get("display") is True

        for (
            call_args
        ) in mock_framework.compare_balanced_class_distributions.call_args_list:
            kwargs = call_args[1]
            assert kwargs.get("display") is True

        for call_args in mock_framework.generate_learning_curves.call_args_list:
            kwargs = call_args[1]
            assert kwargs.get("display") is True

    @patch("balancr.cli.config.load_config")
    @patch("balancr.cli.commands.BalancingFramework")
    def test_format_none_handling(
        self,
        mock_balancing_framework,
        mock_load_config,
        args_run_comparison,
        full_config,
        mock_framework,
    ):
        """Test handling of 'none' format type in save_metrics_formats and save_vis_formats."""
        # Set up configuration with "none" included in both format lists
        config = dict(full_config)
        config["output"]["save_metrics_formats"] = ["csv", "none"]
        config["output"]["save_vis_formats"] = ["png", "none"]

        # Enable cross-validation to test CV-related code paths
        config["evaluation"]["cross_validation"] = 5

        mock_load_config.return_value = config

        # Set up mock framework instance
        mock_balancing_framework.return_value = mock_framework

        # Reset mocks to clear any previous calls
        mock_framework.save_classifier_results.reset_mock()
        mock_framework.inspect_class_distribution.reset_mock()
        mock_framework.compare_balanced_class_distributions.reset_mock()
        mock_framework.generate_learning_curves.reset_mock()

        # Run the test without patching the methods we need to test
        with patch("balancr.cli.commands.plot_comparison_results"):
            # Run comparison
            commands.run_comparison(args_run_comparison)

        # Verify that methods were called
        assert mock_framework.save_classifier_results.called
        assert mock_framework.inspect_class_distribution.called
        assert mock_framework.compare_balanced_class_distributions.called

        # Verify only the non-"none" formats were used in save_classifier_results
        for call in mock_framework.save_classifier_results.call_args_list:
            kwargs = call[1]
            assert kwargs.get("file_type") != "none"

        # Verify only the non-"none" formats were used in save paths
        for call in mock_framework.inspect_class_distribution.call_args_list:
            kwargs = call[1]
            if "save_path" in kwargs:
                assert ".none" not in str(kwargs["save_path"])

        # Ensure the execution completed successfully
        assert mock_framework.train_classifiers.called


class TestResetConfigCommand:
    """Tests for the reset_config command in the balancr CLI."""

    @pytest.fixture
    def args_reset_config(self, mock_config_path):
        """Create mock arguments for reset_config command."""
        args = MagicMock()
        args.config_path = str(mock_config_path)
        return args

    @patch("balancr.cli.config.initialise_config")
    @patch("balancr.cli.commands.logging.info")
    def test_reset_config_success(
        self, mock_info, mock_initialise_config, args_reset_config
    ):
        """Test successful configuration reset."""
        # Call function
        result = commands.reset_config(args_reset_config)

        # Verify initialise_config was called with force=True
        mock_initialise_config.assert_called_once_with(
            args_reset_config.config_path, force=True
        )

        # Verify success message was logged
        mock_info.assert_called_once_with("Configuration has been reset to defaults")

        # Verify success return code
        assert result == 0

    @patch("balancr.cli.config.initialise_config")
    @patch("balancr.cli.commands.logging.error")
    def test_reset_config_exception(
        self, mock_error, mock_initialise_config, args_reset_config
    ):
        """Test handling of exceptions during configuration reset."""
        # Make initialise_config raise an exception
        mock_initialise_config.side_effect = Exception("Permission denied")

        # Call function
        result = commands.reset_config(args_reset_config)

        # Verify error was logged
        mock_error.assert_called_once()
        assert "Failed to reset configuration" in mock_error.call_args[0][0]
        assert "Permission denied" in mock_error.call_args[0][0]

        # Verify error return code
        assert result == 1
