"""
End-to-end tests for the basic workflow of the Balancr framework.
These tests verify that the complete workflow functions correctly from
data loading to results generation.
"""

import os
import pytest
import subprocess
import tempfile
import pandas as pd
import glob
import shutil


def run_command(command, config_path=None):
    """
    Run a Balancr CLI command and return the result.

    Args:
        command (str): The command to run (without 'balancr' prefix)
        config_path (str): Optional path to config file

    Returns:
        str: Command output
        int: Return code
    """
    # Build the command
    if config_path:
        if "--config-path" not in command:
            # Split the command into the subcommand and its arguments
            parts = command.split(maxsplit=1)
            subcommand = parts[0]
            args = parts[1] if len(parts) > 1 else ""

            # Reconstruct with --config-path as part of the subcommand arguments
            command = f"{subcommand} {args} --config-path {config_path}"

    full_command = f"balancr {command}"

    result = subprocess.run(full_command, shell=True, capture_output=True, text=True)
    print(f"Command: {full_command}")
    print(f"Exit code: {result.returncode}")
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")
    return result.stdout, result.returncode


def run_command_list(command, config_path=None):
    """Run command using list of arguments instead of shell string"""
    cmd_parts = command.split()
    cmd_list = ["balancr"] + cmd_parts

    if config_path:
        cmd_list.extend(["--config-path", config_path])

    print(f"Executing command with args: {cmd_list}")
    result = subprocess.run(cmd_list, capture_output=True, text=True)
    print(f"Exit code: {result.returncode}")
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")
    return result.stdout, result.returncode


@pytest.fixture
def test_dataset():
    """
    Create a small test dataset for end-to-end testing.

    Returns:
        Path: Path to the test dataset
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    dataset_path = os.path.join(temp_dir, "test_dataset.csv")

    # Create a simple imbalanced dataset
    def create_synthetic_data(n_majority=20, n_minority=10):
        """Create synthetic data with specified class balance"""
        import numpy as np

        # Create data for majority class (class 0)
        majority_feature1 = np.random.uniform(0, 0.5, n_majority)
        majority_feature2 = np.random.uniform(0, 0.5, n_majority)
        majority_class = np.zeros(n_majority)

        # Create data for minority class (class 1)
        minority_feature1 = np.random.uniform(0.5, 1.0, n_minority)
        minority_feature2 = np.random.uniform(0.5, 1.0, n_minority)
        minority_class = np.ones(n_minority)

        # Combine the data
        feature1 = np.concatenate([majority_feature1, minority_feature1])
        feature2 = np.concatenate([majority_feature2, minority_feature2])
        target = np.concatenate([majority_class, minority_class])

        # Create a dictionary for pandas DataFrame
        data = {"feature1": feature1, "feature2": feature2, "target": target}

        return data

    data = create_synthetic_data(n_majority=20, n_minority=10)

    df = pd.DataFrame(data)
    df.to_csv(dataset_path, index=False)

    yield dataset_path

    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_config():
    """
    Create a temporary configuration file for tests.

    Returns:
        Path: Path to the temporary config file
    """
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, "config.json")

    yield config_path

    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_output_dir():
    """
    Create a temporary output directory for test results.

    Returns:
        Path: Path to the temporary output directory
    """
    temp_dir = tempfile.mkdtemp()

    yield temp_dir

    # Clean up
    shutil.rmtree(temp_dir)


def test_basic_workflow(test_dataset, temp_config, temp_output_dir):
    """
    Test the basic Balancr workflow from data loading to results generation.

    This test:
    1. Loads a simple dataset
    2. Configures minimal preprocessing
    3. Selects a single balancing technique (SMOTE)
    4. Selects a single classifier (RandomForestClassifier)
    5. Runs the comparison
    6. Verifies the results structure and content
    """
    # Step 1: Load the dataset
    output, return_code = run_command(
        f" --config-path {temp_config} load-data {test_dataset} -t target",
        config_path=temp_config,
    )
    assert return_code == 0, "Failed to load data"

    # Step 2: Configure preprocessing
    output, return_code = run_command(
        f" --config-path {temp_config} preprocess --scale standard --handle-missing mean",
        config_path=temp_config,
    )
    assert return_code == 0, "Failed to configure preprocessing"

    # Step 3: Select a balancing technique
    output, return_code = run_command(
        f" --config-path {temp_config} select-techniques SMOTE", config_path=temp_config
    )
    assert return_code == 0, "Failed to select techniques"

    # Step 4: Select a classifier
    output, return_code = run_command(
        f" --config-path {temp_config} select-classifiers RandomForestClassifier",
        config_path=temp_config,
    )
    assert return_code == 0, "Failed to select classifiers"

    # Step 5: Configure metrics
    output, return_code = run_command(
        f" --config-path {temp_config} configure-metrics --metrics precision recall f1",
        config_path=temp_config,
    )
    assert return_code == 0, "Failed to configure metrics"

    # Step 6: Configure visualisations
    output, return_code = run_command(
        f" --config-path {temp_config} configure-visualisations --types distribution metrics --save-formats png "
        "--no-display",
        config_path=temp_config,
    )
    assert return_code == 0, "Failed to configure visualisations"

    # Step 7: Configure evaluation
    output, return_code = run_command(
        f" --config-path {temp_config} configure-evaluation --test-size 0.3 --random-state 42",
        config_path=temp_config,
    )
    assert return_code == 0, "Failed to configure evaluation"

    # Step 8: Run comparison
    output, return_code = run_command(
        f" --config-path {temp_config} run --output-dir {temp_output_dir}",
        config_path=temp_config,
    )
    assert return_code == 0, "Failed to run comparison"

    # Verify results structure
    assert os.path.exists(temp_output_dir), "Output directory does not exist"

    # Check for balanced datasets directory
    balanced_dir = os.path.join(temp_output_dir, "balanced_datasets")
    assert os.path.exists(balanced_dir), "Balanced datasets directory not generated"

    # Check for balanced SMOTE dataset
    smote_file = os.path.join(balanced_dir, "balanced_SMOTE.csv")
    assert os.path.exists(smote_file), "SMOTE balanced dataset not generated"

    # Check that the SMOTE dataset has expected rows
    smote_df = pd.read_csv(smote_file)
    # Should balance classes, so expect roughly equal distribution
    class_counts = smote_df["target"].value_counts()
    assert len(class_counts) == 2, "Expected 2 classes in balanced dataset"
    assert abs(class_counts[0] - class_counts[1]) <= 1, "Classes not balanced properly"

    # Check for RandomForestClassifier directory
    rf_dir = os.path.join(temp_output_dir, "RandomForestClassifier")
    assert os.path.exists(rf_dir), "RandomForestClassifier directory not generated"

    # Check for metrics on original test directory
    metrics_dir = os.path.join(rf_dir, "metrics_on_original_test")
    assert os.path.exists(metrics_dir), "Metrics directory not generated"

    # Check for comparison results CSV
    results_file = os.path.join(metrics_dir, "comparison_results.csv")
    assert os.path.exists(results_file), "Comparison results not generated"

    # Check for metrics comparison visualisation
    metrics_vis = os.path.join(metrics_dir, "metrics_comparison.png")
    assert os.path.exists(metrics_vis), "Metrics visualisation not generated"

    # Check for class distribution visualisations
    imbalanced_vis = glob.glob(
        os.path.join(temp_output_dir, "imbalanced_class_distribution*.png")
    )
    balanced_vis = glob.glob(
        os.path.join(temp_output_dir, "balanced_class_distribution*.png")
    )
    assert (
        len(imbalanced_vis) > 0
    ), "Imbalanced class distribution visualisation not generated"
    assert (
        len(balanced_vis) > 0
    ), "Balanced class distribution visualisation not generated"

    # Verify metrics content
    results_df = pd.read_csv(results_file)
    assert "SMOTE" in results_df.columns, "SMOTE results not found in comparison"

    # Find metric names in the first column
    metric_col = results_df.columns[0]  # Usually "Unnamed: 0"
    metrics_in_file = results_df[metric_col].values

    # Test specific metric values
    assert "f1" in metrics_in_file, "F1 metric not found in results"
    assert "precision" in metrics_in_file, "Precision metric not found in results"
    assert "recall" in metrics_in_file, "Recall metric not found in results"

    # Get index of each metric
    f1_idx = results_df[results_df[metric_col] == "f1"].index[0]
    precision_idx = results_df[results_df[metric_col] == "precision"].index[0]
    recall_idx = results_df[results_df[metric_col] == "recall"].index[0]

    # Verify metric values are in sensible range (0-1)
    assert 0 <= results_df.loc[f1_idx, "SMOTE"] <= 1, "f1 value out of valid range"
    assert (
        0 <= results_df.loc[precision_idx, "SMOTE"] <= 1
    ), "precision value out of valid range"
    assert (
        0 <= results_df.loc[recall_idx, "SMOTE"] <= 1
    ), "recall value out of valid range"
