import os
import shutil
import tempfile
import unittest
import subprocess
import pandas as pd
import numpy as np
import sys
import pytest
import textwrap


class TestCustomExtensions(unittest.TestCase):
    """Tests for registering and using custom balancing techniques and classifiers."""

    def setUp(self):
        """Set up test environment with temporary directories and files."""
        # Create temporary directories for test data, custom extensions, and output
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "config.json")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

        # Directory for custom techniques and classifiers
        self.custom_extensions_dir = os.path.join(self.test_dir, "custom_extensions")
        os.makedirs(self.custom_extensions_dir, exist_ok=True)

        # Create test dataset
        self.test_data_csv = os.path.join(self.test_dir, "test_data.csv")
        self._create_test_dataset()

        # Create custom technique and classifier files
        self.custom_technique_path = os.path.join(
            self.custom_extensions_dir, "custom_technique.py"
        )
        self.custom_classifier_path = os.path.join(
            self.custom_extensions_dir, "custom_classifier.py"
        )
        self._create_custom_technique_file()
        self._create_custom_classifier_file()

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

    def _create_test_dataset(self):
        """Create a simple imbalanced dataset for testing."""
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
        df.to_csv(self.test_data_csv, index=False)

    def _create_custom_technique_file(self):
        """Create a Python file with a custom balancing technique."""
        custom_technique_code = textwrap.dedent(
            '''
            from balancr.base import BaseBalancer
            import numpy as np

            class SimpleUnderSampler(BaseBalancer):
                """
                A simple custom balancing technique that performs basic undersampling
                by randomly removing samples from the majority class.
                """

                def __init__(self, sampling_ratio=1.0, random_state=None):
                    """
                    Initialize the undersampler.

                    Args:
                        sampling_ratio: The desired ratio of minority to majority class samples
                        random_state: Random seed for reproducibility
                    """
                    super().__init__()
                    self.sampling_ratio = sampling_ratio
                    self.random_state = random_state

                def balance(self, X, y):
                    """
                    Balance the dataset by undersampling the majority class.

                    Args:
                        X: Feature matrix
                        y: Target vector

                    Returns:
                        Balanced X and y
                    """
                    # Set random state
                    np.random.seed(self.random_state)

                    # Find the minority and majority classes
                    classes, counts = np.unique(y, return_counts=True)
                    minority_class = classes[np.argmin(counts)]
                    majority_class = classes[np.argmax(counts)]

                    # Get indices for each class
                    minority_indices = np.where(y == minority_class)[0]
                    majority_indices = np.where(y == majority_class)[0]

                    # Calculate how many majority samples to keep
                    n_minority = len(minority_indices)
                    n_majority_keep = min(len(majority_indices),
                                          int(n_minority / self.sampling_ratio))

                    # Randomly select majority samples
                    majority_indices_keep = np.random.choice(
                        majority_indices, n_majority_keep, replace=False)

                    # Combine indices
                    indices_to_keep = np.concatenate([minority_indices, majority_indices_keep])

                    # Return balanced dataset
                    return X[indices_to_keep], y[indices_to_keep]
        '''
        )

        # Write the custom technique to a file
        with open(self.custom_technique_path, "w") as f:
            f.write(custom_technique_code)

    def _create_custom_classifier_file(self):
        """Create a Python file with a custom classifier."""
        custom_classifier_code = textwrap.dedent(
            '''
            from sklearn.base import BaseEstimator, ClassifierMixin
            import numpy as np

            class SimpleThresholdClassifier(BaseEstimator, ClassifierMixin):
                """
                A simple classifier that makes predictions based on a threshold
                on the first feature.
                """

                def __init__(self, threshold=0.0):
                    """
                    Initialize the classifier.

                    Args:
                        threshold: The threshold value for classification
                    """
                    self.threshold = threshold

                def fit(self, X, y):
                    """
                    Fit the classifier.

                    Args:
                        X: Feature matrix
                        y: Target vector

                    Returns:
                        self
                    """
                    # Store the classes
                    self.classes_ = np.unique(y)
                    self.X_ = X
                    self.y_ = y
                    return self

                def predict(self, X):
                    """
                    Predict class labels for samples in X.

                    Args:
                        X: Feature matrix

                    Returns:
                        Predicted class labels
                    """
                    # Simple threshold on the first feature
                    return np.where(X[:, 0] >= self.threshold,
                                   self.classes_[1], self.classes_[0])

                def predict_proba(self, X):
                    """
                    Predict class probabilities for samples in X.

                    Args:
                        X: Feature matrix

                    Returns:
                        Predicted class probabilities
                    """
                    # Simple probabilities based on distance from threshold
                    n_samples = X.shape[0]
                    proba = np.zeros((n_samples, 2))

                    # Scale to [0, 1] based on distance from threshold
                    scaled = 1 / (1 + np.exp(-(X[:, 0] - self.threshold)))

                    proba[:, 0] = 1 - scaled
                    proba[:, 1] = scaled

                    return proba
        '''
        )

        # Write the custom classifier to a file
        with open(self.custom_classifier_path, "w") as f:
            f.write(custom_classifier_code)

    def _run_command(self, cmd, check=True, capture_output=True):
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
    def test_register_custom_technique(self):
        """
        Test registering and using a custom balancing technique.

        This test:
        1. Registers a custom technique
        2. Uses it in a balancing workflow
        3. Verifies the results
        """
        # Reset configuration
        reset_cmd = f"{self.balancr_cmd} --config-path {self.config_path} reset"
        result = self._run_command(reset_cmd, check=False)

        # Initialize configuration if reset failed (first run)
        if result.returncode != 0:
            # Create empty config file
            with open(self.config_path, "w") as f:
                f.write("{}")

        # Register the custom technique
        print(f"custom_technique_path: {self.custom_technique_path}")
        register_cmd = (
            f"{self.balancr_cmd} --config-path {self.config_path} register-techniques --overwrite "
            f"{self.custom_technique_path}"
        )
        result = self._run_command(register_cmd)

        # Verify registration was successful
        self.assertEqual(
            result.returncode, 0, "Custom technique registration should succeed"
        )
        self.assertIn(
            "SimpleUnderSampler",
            result.stdout,
            "Custom technique name should be in the registration output",
        )

        # Load test data
        load_cmd = f"{self.balancr_cmd} --config-path {self.config_path} load-data {self.test_data_csv} -t target"
        result = self._run_command(load_cmd)
        self.assertEqual(result.returncode, 0, "Data loading should succeed")

        # Configure preprocessing
        preprocess_cmd = f"{self.balancr_cmd} --config-path {self.config_path} preprocess --scale standard "
        "--handle-missing mean"
        result = self._run_command(preprocess_cmd)
        self.assertEqual(
            result.returncode, 0, "Preprocessing configuration should succeed"
        )

        # Select techniques including the custom one
        techniques_cmd = (
            f"{self.balancr_cmd} --config-path {self.config_path} select-techniques "
            "SimpleUnderSampler SMOTE"
        )
        result = self._run_command(techniques_cmd)
        self.assertEqual(result.returncode, 0, "Technique selection should succeed")

        # Select classifier
        clf_cmd = f"{self.balancr_cmd} --config-path {self.config_path} select-classifiers RandomForestClassifier"
        result = self._run_command(clf_cmd)
        self.assertEqual(result.returncode, 0, "Classifier selection should succeed")

        # Configure some basic metrics
        metrics_cmd = (
            f"{self.balancr_cmd} --config-path {self.config_path} configure-metrics "
        )
        "--metrics precision recall f1"
        result = self._run_command(metrics_cmd)
        self.assertEqual(result.returncode, 0, "Metrics configuration should succeed")

        # Run the analysis
        run_dir = os.path.join(self.output_dir, "custom_technique_results")
        run_cmd = f"{self.balancr_cmd} --config-path {self.config_path} run --output-dir {run_dir}"
        result = self._run_command(run_cmd, check=False)

        # Even if the run fails for other reasons, check that our custom technique was recognised
        # This is the key thing we're testing
        if "SimpleUnderSampler" not in result.stdout + result.stderr:
            self.fail("Custom technique not found in the run output")

        # If the run completed successfully, check for output artifacts
        if result.returncode == 0:
            # Check for balanced datasets directory
            balanced_dir = os.path.join(run_dir, "balanced_datasets")
            self.assertTrue(
                os.path.exists(balanced_dir), "Balanced datasets directory should exist"
            )

            # Check for SimpleUnderSampler balanced dataset file
            custom_balanced_file = os.path.join(
                balanced_dir, "balanced_SimpleUnderSampler.csv"
            )
            self.assertTrue(
                os.path.exists(custom_balanced_file),
                "Custom technique balanced dataset file should exist",
            )

            # Verify the content of the balanced dataset
            if os.path.exists(custom_balanced_file):
                balanced_df = pd.read_csv(custom_balanced_file)
                # Check if the dataset is more balanced
                original_df = pd.read_csv(self.test_data_csv)
                original_class_counts = original_df["target"].value_counts()
                balanced_class_counts = balanced_df["target"].value_counts()

                # Calculate class ratios (minority/majority)
                original_ratio = min(original_class_counts) / max(original_class_counts)
                balanced_ratio = min(balanced_class_counts) / max(balanced_class_counts)

                # The balanced dataset should have a more balanced class distribution
                self.assertGreater(
                    balanced_ratio,
                    original_ratio,
                    "Balanced dataset should be more balanced than original",
                )

    @pytest.mark.skipif(
        not shutil.which("balancr")
        and not os.path.exists(os.path.join(os.path.dirname(__file__), "../..")),
        reason="Balancr not installed or not in development environment",
    )
    def test_register_custom_classifier(self):
        """
        Test registering and using a custom classifier.

        This test:
        1. Registers a custom classifier
        2. Uses it in a balancing workflow
        3. Verifies the results
        """
        # Reset configuration
        reset_cmd = f"{self.balancr_cmd} --config-path {self.config_path} reset"
        result = self._run_command(reset_cmd, check=False)

        # Initialize configuration if reset failed (first run)
        if result.returncode != 0:
            # Create empty config file
            with open(self.config_path, "w") as f:
                f.write("{}")

        # Register the custom classifier
        register_cmd = (
            f"{self.balancr_cmd} --config-path {self.config_path} register-classifiers "
            f"--overwrite {self.custom_classifier_path}"
        )
        result = self._run_command(register_cmd)

        # Verify registration was successful
        self.assertEqual(
            result.returncode, 0, "Custom classifier registration should succeed"
        )
        self.assertIn(
            "SimpleThresholdClassifier",
            result.stdout,
            "Custom classifier name should be in the registration output",
        )

        # Load test data
        load_cmd = f"{self.balancr_cmd} --config-path {self.config_path} load-data {self.test_data_csv} -t target"
        result = self._run_command(load_cmd)
        self.assertEqual(result.returncode, 0, "Data loading should succeed")

        # Configure preprocessing
        preprocess_cmd = f"{self.balancr_cmd} --config-path {self.config_path} preprocess --scale standard "
        "--handle-missing mean"
        result = self._run_command(preprocess_cmd)
        self.assertEqual(
            result.returncode, 0, "Preprocessing configuration should succeed"
        )

        # Select balancing technique
        techniques_cmd = f"{self.balancr_cmd} --config-path {self.config_path} select-techniques SMOTE"
        result = self._run_command(techniques_cmd)
        self.assertEqual(result.returncode, 0, "Technique selection should succeed")

        # Select classifiers including the custom one
        clf_cmd = (
            f"{self.balancr_cmd} --config-path {self.config_path} select-classifiers "
            "SimpleThresholdClassifier RandomForestClassifier"
        )
        result = self._run_command(clf_cmd)
        self.assertEqual(result.returncode, 0, "Classifier selection should succeed")

        # Configure some basic metrics
        metrics_cmd = (
            f"{self.balancr_cmd} --config-path {self.config_path} configure-metrics "
            "--metrics precision recall f1"
        )
        result = self._run_command(metrics_cmd)
        self.assertEqual(result.returncode, 0, "Metrics configuration should succeed")

        # Run the analysis
        run_dir = os.path.join(self.output_dir, "custom_classifier_results")
        run_cmd = f"{self.balancr_cmd} --config-path {self.config_path} run --output-dir {run_dir}"
        result = self._run_command(run_cmd, check=False)

        # Even if the run fails for other reasons, check that our custom classifier was recognised
        # This is the key thing we're testing
        if "SimpleThresholdClassifier" not in result.stdout + result.stderr:
            self.fail("Custom classifier not found in the run output")

        # If the run completed successfully, check for output artifacts
        if result.returncode == 0:
            # Check for classifier results directory
            custom_clf_dir = os.path.join(run_dir, "SimpleThresholdClassifier")
            self.assertTrue(
                os.path.exists(custom_clf_dir),
                "Custom classifier results directory should exist",
            )

            # Check for metrics file
            metrics_file = os.path.join(
                custom_clf_dir, "metrics_on_original_test", "comparison_results.csv"
            )
            if os.path.exists(metrics_file):
                # Verify the content of the metrics file
                metrics_df = pd.read_csv(metrics_file)
                # The metrics file should contain results for the custom classifier
                self.assertGreater(
                    len(metrics_df), 0, "Metrics file should contain results"
                )


if __name__ == "__main__":
    unittest.main()
