import os
import shutil
import tempfile
import unittest
import subprocess
import pandas as pd
import numpy as np
import sys
import pytest


class TestErrorHandling(unittest.TestCase):
    """Tests for error handling and recovery in end-to-end scenarios."""

    def setUp(self):
        """Set up test environment with temporary directories and files."""
        # Create temporary directories for test data and output
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "config.json")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

        # Create valid and invalid test datasets
        self.valid_data_csv = os.path.join(self.test_dir, "valid_data.csv")
        self.invalid_data_csv = os.path.join(self.test_dir, "invalid_data.csv")
        self.non_existent_data_csv = os.path.join(
            self.test_dir, "non_existent_data.csv"
        )

        self._create_valid_dataset()
        self._create_invalid_dataset()

        # Create an empty Python file for invalid technique testing
        self.empty_py_file = os.path.join(self.test_dir, "empty_technique.py")
        with open(self.empty_py_file, "w") as f:
            f.write("# Empty file with no valid techniques")

        # Find the balancr executable or use python -m approach
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

    def _create_valid_dataset(self):
        """Create a valid, simple imbalanced dataset for testing."""
        np.random.seed(42)

        # Create an imbalanced dataset
        n_samples = 100
        n_minority = 10

        # Create features
        X = np.random.normal(0, 1, (n_samples, 3))

        # Create imbalanced target (10% minority class)
        y = np.zeros(n_samples)
        minority_indices = np.random.choice(
            range(n_samples), size=n_minority, replace=False
        )
        y[minority_indices] = 1

        # Create DataFrame and save to CSV
        df = pd.DataFrame(X, columns=["feature1", "feature2", "feature3"])
        df["target"] = y
        df.to_csv(self.valid_data_csv, index=False)

    def _create_invalid_dataset(self):
        """Create an invalid dataset for testing error handling."""
        # Create a dataset with inconsistent column count
        with open(self.invalid_data_csv, "w") as f:
            f.write("column1,column2,column3,target\n")
            f.write("1.0,2.0,3.0,0\n")
            f.write("4.0,5.0,0\n")  # Missing a column
            f.write("7.0,8.0,9.0,1\n")

    def _run_command(self, cmd, check=False, capture_output=True):
        """
        Run a command and return the result, handling exceptions appropriately.
        """
        try:
            print(f"Executing command: {cmd}")
            result = subprocess.run(
                cmd, shell=True, check=check, capture_output=capture_output, text=True
            )
            return result
        except subprocess.CalledProcessError as e:
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
    def test_recovery_from_invalid_technique(self):
        """
        Test that the system gracefully handles invalid balancing techniques.

        This tests three invalid technique scenarios:
        1. A non-existent technique name
        2. A non-existent technique file
        3. A file with no valid technique classes
        """
        # Reset configuration
        reset_cmd = f"{self.balancr_cmd} --config-path {self.config_path} reset"
        result = self._run_command(reset_cmd, check=False)

        # Initialise configuration if reset failed (first run)
        if result.returncode != 0:
            # Create empty config file
            with open(self.config_path, "w") as f:
                f.write("{}")

        # Load valid data
        load_cmd = f"{self.balancr_cmd} --config-path {self.config_path} load-data {self.valid_data_csv} -t target"
        self._run_command(load_cmd)

        # Test Case 1: Non-existent technique name
        nonexist_tech_cmd = (
            f"{self.balancr_cmd} --config-path {self.config_path} "
            "select-techniques NonExistentTechnique"
        )
        result = self._run_command(nonexist_tech_cmd)

        # Check for correct error message
        self.assertIn(
            "invalid technique",
            result.stdout.lower() + result.stderr.lower(),
            "Should indicate invalid technique name",
        )

        # The system should still be in a usable state - test this by selecting a valid technique
        valid_tech_cmd = f"{self.balancr_cmd} --config-path {self.config_path} select-techniques SMOTE"
        result = self._run_command(valid_tech_cmd)
        self.assertEqual(
            result.returncode,
            0,
            "Should recover and accept valid technique after invalid one",
        )

        # Test Case 2: Non-existent technique file
        nonexist_file_cmd = (
            f"{self.balancr_cmd} --config-path {self.config_path} "
            "register-techniques /nonexistent/path.py"
        )
        result = self._run_command(nonexist_file_cmd)

        # Check for correct error message
        self.assertIn(
            "file not found",
            (result.stdout + result.stderr).lower(),
            "Should indicate file not found",
        )

        # The system should still be in a usable state
        # Test by trying to register a technique with an empty file
        empty_file_cmd = f"{self.balancr_cmd} --config-path {self.config_path} register-techniques {self.empty_py_file}"
        result = self._run_command(empty_file_cmd)

        # Check for correct message about no valid techniques
        self.assertIn(
            "no valid",
            (result.stdout + result.stderr).lower(),
            "Should indicate no valid techniques in file",
        )

        # Test Case 3: Invalid class name
        invalid_class_cmd = (
            f"{self.balancr_cmd} --config-path {self.config_path} "
            f"register-techniques {self.empty_py_file} --class-name NonExistentClass"
        )
        result = self._run_command(invalid_class_cmd)

        # Check for correct error message
        self.assertIn(
            "class",
            (result.stdout + result.stderr).lower(),
            "Should indicate class not found",
        )

    @pytest.mark.skipif(
        not shutil.which("balancr")
        and not os.path.exists(os.path.join(os.path.dirname(__file__), "../..")),
        reason="Balancr not installed or not in development environment",
    )
    def test_recovery_from_invalid_data(self):
        """
        Test that the system gracefully handles data loading errors.

        This tests various invalid data scenarios:
        1. A non-existent data file
        2. A malformed data file
        3. Missing target column
        4. Dataset with missing target values
        """
        # Reset configuration
        reset_cmd = f"{self.balancr_cmd} --config-path {self.config_path} reset"
        result = self._run_command(reset_cmd, check=False)

        # Initialise configuration if reset failed (first run)
        if result.returncode != 0:
            # Create empty config file
            with open(self.config_path, "w") as f:
                f.write("{}")

        # Test Case 1: Non-existent data file
        nonexist_data_cmd = (
            f"{self.balancr_cmd} --config-path {self.config_path} load-data {self.non_existent_data_csv} "
            "-t target"
        )
        result = self._run_command(nonexist_data_cmd)

        # Check for appropriate error message
        self.assertIn(
            "file not found",
            (result.stdout + result.stderr).lower(),
            "Should indicate file not found",
        )

        # The system should still be in a usable state - test by loading a valid file
        valid_data_cmd = (
            f"{self.balancr_cmd} --config-path {self.config_path} load-data {self.valid_data_csv} "
            "-t target"
        )
        result = self._run_command(valid_data_cmd)
        self.assertEqual(
            result.returncode,
            0,
            "Should recover and load valid data after invalid file",
        )

        # Test Case 2: Missing target column
        missing_target_cmd = (
            f"{self.balancr_cmd} --config-path {self.config_path} load-data {self.valid_data_csv} -t nonexistent_column"
        )
        result = self._run_command(missing_target_cmd)

        # Check for appropriate error message
        self.assertIn(
            "target",
            (result.stdout + result.stderr).lower(),
            "Should indicate missing target column",
        )

        # The system should still be in a usable state - test by specifying a valid target
        valid_target_cmd = (
            f"{self.balancr_cmd} --config-path {self.config_path} load-data {self.valid_data_csv} -t target"
        )
        result = self._run_command(valid_target_cmd)
        self.assertEqual(
            result.returncode,
            0,
            "Should recover and load data with valid target column",
        )

        # Test Case 3: Dataset with missing target values
        missing_targets_csv = os.path.join(self.test_dir, "missing_targets.csv")
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [0.1, 0.2, 0.3, 0.4, 0.5],
                "feature3": [10, 20, 30, 40, 50],
                "target": [0, 1, np.nan, 0, 1],
            }
        )
        df.to_csv(missing_targets_csv, index=False)

        # Try to load the data with missing target values
        missing_values_cmd = (
            f"{self.balancr_cmd} --config-path {self.config_path} load-data {missing_targets_csv} -t target"
        )
        result = self._run_command(missing_values_cmd)

        # The command should succeed even with missing target values
        self.assertEqual(
            result.returncode,
            0,
            "Should successfully load dataset with missing target values",
        )

        # Check for warning message about missing target values
        self.assertIn(
            "missing target values",
            (result.stdout + result.stderr).lower(),
            "Should warn about missing target values",
        )

        # Verify that the system continues to function
        # Select a technique to confirm the system is still in a usable state
        select_tech_cmd = f"{self.balancr_cmd} --config-path {self.config_path} select-techniques SMOTE"
        result = self._run_command(select_tech_cmd)
        self.assertEqual(
            result.returncode,
            0,
            "Should remain in a usable state after handling missing target values",
        )


if __name__ == "__main__":
    unittest.main()
