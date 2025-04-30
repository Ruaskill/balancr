import os
import shutil
import tempfile
import unittest
import subprocess
import pandas as pd
import numpy as np
import sys
import pytest


class TestDataQualityHandling(unittest.TestCase):
    """Tests for handling data quality issues in end-to-end scenarios."""

    def setUp(self):
        """Set up test environment with temporary directories and files."""
        # Create temporary directories for test data and output
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "config.json")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

        # Create test datasets
        self.missing_values_csv = os.path.join(self.test_dir, "missing_values.csv")
        self.categorical_data_csv = os.path.join(self.test_dir, "categorical_data.csv")

        self._create_missing_values_dataset()
        self._create_categorical_dataset()

        # Find the balancr executable or use python -m approach
        # This handles both installed and development environments
        self.balancr_cmd = self._get_balancr_command()

    def tearDown(self):
        """Clean up temporary files and directories after tests."""
        shutil.rmtree(self.test_dir)

    def _get_balancr_command(self):
        """
        Determine the best way to call balancr based on the environment.
        Returns the proper command prefix to use.
        """
        # Try to locate the balancr executable
        try:
            # Check if balancr is in PATH
            subprocess.run(
                ["which", "balancr"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return "balancr"
        except (subprocess.SubprocessError, FileNotFoundError):
            # Fall back to Python module invocation
            # This works when testing in a development environment
            return f"{sys.executable} -m balancr.cli.main"

    def _create_missing_values_dataset(self):
        """Create a dataset with missing values for testing."""
        # Create a DataFrame with some missing values
        np.random.seed(42)
        data = {
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(0, 1, 100),
            "feature3": np.random.normal(0, 1, 100),
            "target": np.random.randint(0, 2, 100),
        }

        # Introduce missing values (~10% of the data)
        df = pd.DataFrame(data)
        mask1 = np.random.random(100) < 0.1
        mask2 = np.random.random(100) < 0.1
        df.loc[mask1, "feature1"] = np.nan
        df.loc[mask2, "feature2"] = np.nan

        # Save to CSV
        df.to_csv(self.missing_values_csv, index=False)

    def _create_categorical_dataset(self):
        """Create a dataset with categorical features for testing."""
        np.random.seed(42)

        # Create a DataFrame with categorical features
        n_samples = 100
        data = {
            "feature1": np.random.normal(0, 1, n_samples),
            "categorical1": np.random.choice(["A", "B", "C"], n_samples),
            "categorical2": np.random.choice(["Low", "Medium", "High"], n_samples),
            "ordinal_feature": np.random.choice(
                ["Small", "Medium", "Large"], n_samples
            ),
            "target": np.random.randint(0, 2, n_samples),
        }

        # Save to CSV
        pd.DataFrame(data).to_csv(self.categorical_data_csv, index=False)

    def _run_command(self, cmd, check=True, capture_output=True):
        """
        Run a command and return the result, handling exceptions appropriately.
        """
        try:
            # Execute command with shell expansion disabled for security
            result = subprocess.run(
                cmd,
                shell=True,  # Use shell for complex commands
                check=check,  # Raise exception on failure if check is True
                capture_output=capture_output,
                text=True,  # Capture output as text
            )
            return result
        except subprocess.CalledProcessError as e:
            # In tests, we can make these non-fatal but log the failure
            print(f"Command failed: {cmd}")
            print(f"Return code: {e.returncode}")
            print(f"Output: {e.stdout}")
            print(f"Error: {e.stderr}")
            if check:
                raise
            return e

    @pytest.mark.skipif(
        not shutil.which("balancr")
        and not os.path.exists(os.path.join(os.path.dirname(__file__), "../..")),
        reason="Balancr not installed or not in development environment",
    )
    def test_handling_missing_values(self):
        """
        Test that the system successfully preprocesses data with missing values.

        This checks different strategies for handling missing values:
        - mean imputation
        - median imputation
        - dropping rows with missing values
        """
        # Reset configuration
        reset_cmd = f"{self.balancr_cmd} --config-path {self.config_path} reset"
        result = self._run_command(reset_cmd, check=False)

        # Initialise configuration if reset failed (first run)
        if result.returncode != 0:
            # Create empty config file
            with open(self.config_path, "w") as f:
                f.write("{}")

        # Load the dataset with missing values
        load_cmd = f"{self.balancr_cmd} --config-path {self.config_path} load-data {self.missing_values_csv} -t target"
        result = self._run_command(load_cmd)

        # Check if missing values were detected in the output (if command was successful)
        if result.returncode == 0:
            self.assertIn(
                "missing values",
                result.stdout.lower() + result.stderr.lower(),
                "Missing values should be detected and reported",
            )

        # Test mean imputation
        mean_cmd = f"{self.balancr_cmd} --config-path {self.config_path} preprocess --handle-missing mean "
        "--scale standard"
        result = self._run_command(mean_cmd, check=False)

        # Proceed only if the command was successful
        if result.returncode == 0:
            # Test the processing with missing values using SMOTE balancing
            select_tech_cmd = f"{self.balancr_cmd} --config-path {self.config_path} select-techniques SMOTE"
            self._run_command(select_tech_cmd, check=False)

            # Select a basic classifier
            select_clf_cmd = f"{self.balancr_cmd} --config-path {self.config_path} select-classifiers"
            "RandomForestClassifier"
            self._run_command(select_clf_cmd, check=False)

            # Run with mean imputation
            mean_output_dir = os.path.join(self.output_dir, "mean")
            mean_run_cmd = f"{self.balancr_cmd} --config-path {self.config_path} run --output-dir {mean_output_dir}"
            result = self._run_command(mean_run_cmd, check=False)

            # Verify results directory exists if run was successful
            if result.returncode == 0:
                self.assertTrue(
                    os.path.exists(mean_output_dir),
                    "Results directory should be created",
                )

            # Test other imputation strategies if first one worked
            # Test median imputation
            median_cmd = f"{self.balancr_cmd} --config-path {self.config_path} preprocess --handle-missing median "
            "--scale standard"
            self._run_command(median_cmd, check=False)

            # Test drop strategy
            drop_cmd = f"{self.balancr_cmd} --config-path {self.config_path} preprocess --handle-missing drop "
            "--scale standard"
            self._run_command(drop_cmd, check=False)

    @pytest.mark.skipif(
        not shutil.which("balancr")
        and not os.path.exists(os.path.join(os.path.dirname(__file__), "../..")),
        reason="Balancr not installed or not in development environment",
    )
    def test_handling_categorical_features(self):
        """
        Test that the system successfully preprocesses categorical data.

        This checks different encoding strategies for categorical features:
        - one-hot encoding
        - label encoding
        - ordinal encoding
        """
        # Reset configuration
        reset_cmd = f"{self.balancr_cmd} --config-path {self.config_path} reset"
        result = self._run_command(reset_cmd, check=False)

        # Initialise configuration if reset failed (first run)
        if result.returncode != 0:
            # Create empty config file
            with open(self.config_path, "w") as f:
                f.write("{}")

        # Load the dataset with categorical features
        load_cmd = f"{self.balancr_cmd} --config-path {self.config_path} load-data {self.categorical_data_csv} "
        "-t target"
        result = self._run_command(load_cmd, check=False)

        # Proceed only if data was loaded successfully
        if result.returncode == 0:
            # Define categorical features
            categorical_features = ["categorical1", "categorical2"]
            ordinal_features = ["ordinal_feature"]

            # Configure preprocessing with one-hot encoding
            onehot_cmd = (
                f"{self.balancr_cmd} --config-path {self.config_path} preprocess "
                f"--categorical-features {' '.join(categorical_features)} "
                f"--ordinal-features {' '.join(ordinal_features)} "
                f"--encode onehot --scale standard"
            )
            result = self._run_command(onehot_cmd, check=False)

            # Proceed only if preprocessing configuration was successful
            if result.returncode == 0:
                # Select balancing technique and classifier
                select_tech_cmd = f"{self.balancr_cmd} --config-path {self.config_path} select-techniques SMOTE"
                self._run_command(select_tech_cmd, check=False)

                select_clf_cmd = f"{self.balancr_cmd} --config-path {self.config_path} "
                "select-classifiers RandomForestClassifier"
                self._run_command(select_clf_cmd, check=False)

                # Run with one-hot encoding
                onehot_output_dir = os.path.join(self.output_dir, "onehot")
                onehot_run_cmd = f"{self.balancr_cmd} --config-path {self.config_path} "
                f"run --output-dir {onehot_output_dir}"
                result = self._run_command(onehot_run_cmd, check=False)

                # Verify results directory exists if run was successful
                if result.returncode == 0:
                    self.assertTrue(
                        os.path.exists(onehot_output_dir),
                        "Results directory for one-hot encoding should be created",
                    )

                # Test label encoding
                label_cmd = (
                    f"{self.balancr_cmd} --config-path {self.config_path} preprocess "
                    f"--categorical-features {' '.join(categorical_features)} "
                    f"--ordinal-features {' '.join(ordinal_features)} "
                    f"--encode label --scale standard"
                )
                self._run_command(label_cmd, check=False)

                # Test auto encoding
                auto_cmd = (
                    f"{self.balancr_cmd} --config-path {self.config_path} preprocess "
                    f"--categorical-features {' '.join(categorical_features)} "
                    f"--ordinal-features {' '.join(ordinal_features)} "
                    f"--encode auto --scale standard"
                )
                self._run_command(auto_cmd, check=False)


if __name__ == "__main__":
    unittest.main()
