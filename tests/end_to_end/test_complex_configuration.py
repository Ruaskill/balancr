"""
End-to-end tests for complex configurations in the Balancr framework.
These tests verify the framework can handle multiple techniques, classifiers,
and configuration options simultaneously.
"""

import os
import pytest
import subprocess
import tempfile
import pandas as pd
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


@pytest.fixture
def test_dataset_categorical():
    """
    Create a more complex test dataset with categorical features.

    Returns:
        Path: Path to the test dataset
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    dataset_path = os.path.join(temp_dir, "test_complex_dataset.csv")

    # Create an imbalanced dataset with categorical features
    def create_complex_data(n_majority=80, n_minority=20):
        """Create synthetic data with numerical and categorical features"""
        import numpy as np

        # Create numerical features
        majority_feature1 = np.random.uniform(0, 0.5, n_majority)
        majority_feature2 = np.random.uniform(0, 0.5, n_majority)

        minority_feature1 = np.random.uniform(0.5, 1.0, n_minority)
        minority_feature2 = np.random.uniform(0.5, 1.0, n_minority)

        # Create categorical features
        categories = ["A", "B", "C", "D"]
        majority_cat1 = np.random.choice(categories, n_majority)
        majority_cat2 = np.random.choice(["Low", "Medium", "High"], n_majority)

        minority_cat1 = np.random.choice(categories, n_minority)
        minority_cat2 = np.random.choice(["Low", "Medium", "High"], n_minority)

        # Target classes
        majority_class = np.zeros(n_majority)
        minority_class = np.ones(n_minority)

        # Combine the data
        feature1 = np.concatenate([majority_feature1, minority_feature1])
        feature2 = np.concatenate([majority_feature2, minority_feature2])
        categorical1 = np.concatenate([majority_cat1, minority_cat1])
        categorical2 = np.concatenate([majority_cat2, minority_cat2])
        target = np.concatenate([majority_class, minority_class])

        # Introduce some missing values (approximately 5%)
        mask = np.random.random(len(feature1)) < 0.05
        feature1[mask] = np.nan

        mask = np.random.random(len(feature2)) < 0.05
        feature2[mask] = np.nan

        mask = np.random.random(len(categorical1)) < 0.05
        categorical1[mask] = None

        mask = np.random.random(len(categorical2)) < 0.05
        categorical2[mask] = None

        # Create a dictionary for pandas DataFrame
        data = {
            "numerical_feature1": feature1,
            "numerical_feature2": feature2,
            "categorical_feature1": categorical1,
            "categorical_feature2": categorical2,
            "target": target,
        }

        return data

    data = create_complex_data(n_majority=80, n_minority=20)

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


def test_complex_configuration(test_dataset_categorical, temp_config, temp_output_dir):
    """
    Test Balancr with a complex configuration including:
    - Multiple balancing techniques (SMOTE, ADASYN)
    - Multiple classifiers (RandomForestClassifier, LogisticRegression)
    - Cross-validation enabled
    - Categorical feature preprocessing
    - All visualisation types
    """
    # Step 1: Load the dataset
    output, return_code = run_command(
        f"--config-path {temp_config} load-data {test_dataset_categorical} -t target",
        config_path=temp_config,
    )
    assert return_code == 0, "Failed to load data"

    # Step 2: Configure preprocessing with categorical features
    output, return_code = run_command(
        f"--config-path {temp_config} preprocess --scale standard --handle-missing mean --encode auto "
        f"--categorical-features categorical_feature1 categorical_feature2 "
        f"--handle-constant-features drop",
        config_path=temp_config,
    )
    assert return_code == 0, "Failed to configure preprocessing"

    # Step 3: Select multiple balancing techniques
    output, return_code = run_command(
        f"--config-path {temp_config} select-techniques SMOTE ADASYN",
        config_path=temp_config,
    )
    assert return_code == 0, "Failed to select techniques"

    # Step 4: Select multiple classifiers
    output, return_code = run_command(
        f"--config-path {temp_config} select-classifiers RandomForestClassifier LogisticRegression",
        config_path=temp_config,
    )
    assert return_code == 0, "Failed to select classifiers"

    # Step 5: Configure metrics
    output, return_code = run_command(
        f"--config-path {temp_config} configure-metrics --metrics precision recall f1 roc_auc g_mean",
        config_path=temp_config,
    )
    assert return_code == 0, "Failed to configure metrics"

    # Step 6: Configure visualisations
    output, return_code = run_command(
        f"--config-path {temp_config} configure-visualisations --types all --save-formats png pdf --no-display",
        config_path=temp_config,
    )
    assert return_code == 0, "Failed to configure visualisations"

    # Step 7: Configure evaluation with cross-validation
    output, return_code = run_command(
        f"--config-path {temp_config} configure-evaluation --test-size 0.3 --cross-validation 3 --random-state 42 "
        f"--learning-curve-folds 3  --learning-curve-points 3",
        config_path=temp_config,
    )
    assert return_code == 0, "Failed to configure evaluation"

    # Step 8: Run comparison
    output, return_code = run_command(
        f"--config-path {temp_config} run --output-dir {temp_output_dir}",
        config_path=temp_config,
    )
    assert return_code == 0, "Failed to run comparison"

    # Verify results structure
    assert os.path.exists(temp_output_dir), "Output directory does not exist"

    # Check for balanced datasets directory
    balanced_dir = os.path.join(temp_output_dir, "balanced_datasets")
    assert os.path.exists(balanced_dir), "Balanced datasets directory not generated"

    # Check for balanced datasets for all techniques
    for technique in ["SMOTE", "ADASYN"]:
        technique_file = os.path.join(balanced_dir, f"balanced_{technique}.csv")
        assert os.path.exists(
            technique_file
        ), f"{technique} balanced dataset not generated"

        # Check that the dataset has expected rows
        df = pd.read_csv(technique_file)
        # Should balance classes, so expect roughly equal distribution
        class_counts = df["target"].value_counts()
        assert len(class_counts) == 2, f"Expected 2 classes in {technique} dataset"

    # Check for classifier directories
    for classifier in ["RandomForestClassifier", "LogisticRegression"]:
        clf_dir = os.path.join(temp_output_dir, classifier)
        assert os.path.exists(clf_dir), f"{classifier} directory not generated"

        # Check for metrics on original test directory
        metrics_dir = os.path.join(clf_dir, "metrics_on_original_test")
        assert os.path.exists(
            metrics_dir
        ), f"Metrics directory for {classifier} not generated"

        # Check for comparison results CSV
        results_file = os.path.join(metrics_dir, "comparison_results.csv")
        assert os.path.exists(
            results_file
        ), f"Comparison results for {classifier} not generated"

        # Check for metrics visualisations
        metrics_vis = os.path.join(metrics_dir, "metrics_comparison.png")
        assert os.path.exists(
            metrics_vis
        ), f"Metrics visualisation for {classifier} not generated"

        # Check for learning curves
        learning_curves = os.path.join(metrics_dir, "learning_curves.png")
        assert os.path.exists(
            learning_curves
        ), f"Learning curves for {classifier} not generated"

        # Check for cross-validation metrics directory
        cv_metrics_dir = os.path.join(clf_dir, "metrics_on_balanced_cv")
        assert os.path.exists(
            cv_metrics_dir
        ), f"CV metrics directory for {classifier} not generated"

        # Check for CV results
        cv_results_file = os.path.join(cv_metrics_dir, "comparison_results.csv")
        assert os.path.exists(
            cv_results_file
        ), f"CV comparison results for {classifier} not generated"

    # Check for radar charts
    for classifier in ["RandomForestClassifier", "LogisticRegression"]:
        clf_dir = os.path.join(temp_output_dir, classifier)
        standard_radar = os.path.join(clf_dir, "metrics_on_original_test_radar.png")
        cv_radar = os.path.join(clf_dir, "cv_metrics_radar.png")
        assert os.path.exists(
            standard_radar
        ), f"Standard metrics radar chart for {classifier} not generated"
        assert os.path.exists(
            cv_radar
        ), f"CV metrics radar chart for {classifier} not generated"

    # Check for 3D plots
    assert os.path.exists(
        os.path.join(temp_output_dir, "metrics_on_original_test_3d.html")
    ), "3D plot for standard metrics not generated"
    assert os.path.exists(
        os.path.join(temp_output_dir, "cv_metrics_3d.html")
    ), "3D plot for CV metrics not generated"

    # Verify metrics content for both classifiers
    for classifier in ["RandomForestClassifier", "LogisticRegression"]:
        # Check standard metrics
        results_file = os.path.join(
            temp_output_dir,
            classifier,
            "metrics_on_original_test",
            "comparison_results.csv",
        )
        results_df = pd.read_csv(results_file)

        # Check all techniques are in the results
        for technique in ["SMOTE", "ADASYN"]:
            assert (
                technique in results_df.columns
            ), f"{technique} results not found for {classifier}"

        # Find metric names in the first column
        metric_col = results_df.columns[0]  # Usually "Unnamed: 0"
        metrics_in_file = results_df[metric_col].values

        # Test all requested metrics are present
        for metric in ["f1", "precision", "recall", "roc_auc", "g_mean"]:
            assert (
                metric in metrics_in_file
            ), f"{metric} not found in results for {classifier}"

            # Get index of the metric
            metric_idx = results_df[results_df[metric_col] == metric].index[0]

            # Check values for each technique
            for technique in ["SMOTE", "ADASYN"]:
                metric_value = results_df.loc[metric_idx, technique]
                # Some metrics might be NaN if the classifier couldn't calculate them
                # But non-NaN values should be in the valid range
                if not pd.isna(metric_value):
                    assert (
                        0 <= metric_value <= 1
                    ), f"{metric} value for {technique} out of valid range"

        # Check CV metrics
        cv_results_file = os.path.join(
            temp_output_dir,
            classifier,
            "metrics_on_balanced_cv",
            "comparison_results.csv",
        )
        cv_results_df = pd.read_csv(cv_results_file)

        # Check all techniques are in the CV results
        for technique in ["SMOTE", "ADASYN"]:
            assert (
                technique in cv_results_df.columns
            ), f"{technique} CV results not found for {classifier}"

        # Check CV metrics include means and standard deviations
        cv_metric_col = cv_results_df.columns[0]
        cv_metrics_in_file = cv_results_df[cv_metric_col].values

        # Test for CV metrics (with _mean suffix)
        for metric in ["cv_f1_mean", "cv_precision_mean", "cv_recall_mean"]:
            assert (
                metric in cv_metrics_in_file
            ), f"{metric} not found in CV results for {classifier}"


def test_multiple_techniques_workflow(
    test_dataset_categorical, temp_config, temp_output_dir
):
    """
    Test Balancr with multiple balancing techniques.

    This test focuses specifically on the comparison of different balancing techniques:
    - Multiple undersampling techniques (RandomUnderSampler, NearMiss)
    - Multiple oversampling techniques (SMOTE, ADASYN)
    - Combination techniques (SMOTETomek)
    - Comparing the metrics across all techniques
    """
    # Step 1: Load the dataset
    output, return_code = run_command(
        f" --config-path {temp_config} load-data {test_dataset_categorical} -t target",
    )
    assert return_code == 0, "Failed to load data"

    # Step 2: Configure preprocessing (simple config for this test)
    output, return_code = run_command(
        f" --config-path {temp_config} preprocess --scale standard --handle-missing mean "
        "--categorical-features categorical_feature1 categorical_feature2"
    )
    assert return_code == 0, "Failed to configure preprocessing"

    # Step 3: Select multiple balancing techniques from different categories
    output, return_code = run_command(
        f" --config-path {temp_config} select-techniques SMOTE ADASYN RandomUnderSampler NearMiss SMOTETomek",
    )
    assert return_code == 0, "Failed to select techniques"

    # Step 4: Select a single classifier for simplicity
    output, return_code = run_command(
        f" --config-path {temp_config} select-classifiers RandomForestClassifier",
    )
    assert return_code == 0, "Failed to select classifier"

    # Step 5: Configure metrics focusing on imbalanced data metrics
    output, return_code = run_command(
        f" --config-path {temp_config} configure-metrics --metrics precision recall f1 g_mean specificity",
    )
    assert return_code == 0, "Failed to configure metrics"

    # Step 6: Enable visualisations to compare technique performance
    output, return_code = run_command(
        f" --config-path {temp_config} configure-visualisations --types metrics distribution --save-formats png "
        "--no-display",
    )
    assert return_code == 0, "Failed to configure visualisations"

    # Step 7: Run comparison
    output, return_code = run_command(
        f" --config-path {temp_config} run --output-dir {temp_output_dir}",
    )
    assert return_code == 0, "Failed to run comparison"

    # Verify results
    assert os.path.exists(temp_output_dir), "Output directory does not exist"

    # Check that all technique datasets were generated
    balanced_dir = os.path.join(temp_output_dir, "balanced_datasets")
    assert os.path.exists(balanced_dir), "Balanced datasets directory not generated"

    for technique in [
        "SMOTE",
        "ADASYN",
        "RandomUnderSampler",
        "NearMiss",
        "SMOTETomek",
    ]:
        technique_file = os.path.join(balanced_dir, f"balanced_{technique}.csv")
        assert os.path.exists(
            technique_file
        ), f"{technique} balanced dataset not generated"

        # Verify the datasets have different characteristics
        df = pd.read_csv(technique_file)

        # Each technique should have produced a balanced dataset
        # But with different characteristics
        if technique in ["RandomUnderSampler", "NearMiss"]:
            # Undersampling should reduce the majority class
            assert len(df) < 100, f"{technique} should have reduced dataset size"
        elif technique in ["SMOTE", "ADASYN"]:
            # Oversampling should increase the minority class
            assert len(df) >= 100, f"{technique} should have increased dataset size"

    # Check for class distribution comparison visualisation
    assert os.path.exists(
        os.path.join(temp_output_dir, "balanced_class_distribution.png")
    ), "Class distribution comparison not generated"

    # Check for metrics comparison
    metrics_dir = os.path.join(
        temp_output_dir, "RandomForestClassifier", "metrics_on_original_test"
    )
    assert os.path.exists(metrics_dir), "Metrics directory not generated"

    # Verify metrics results contain all techniques
    results_file = os.path.join(metrics_dir, "comparison_results.csv")
    assert os.path.exists(results_file), "Comparison results not generated"

    # Load and verify metrics content
    results_df = pd.read_csv(results_file)
    for technique in [
        "SMOTE",
        "ADASYN",
        "RandomUnderSampler",
        "NearMiss",
        "SMOTETomek",
    ]:
        assert (
            technique in results_df.columns
        ), f"{technique} missing from metrics comparison"


def test_multiple_classifiers_workflow(
    test_dataset_categorical, temp_config, temp_output_dir
):
    """
    Test Balancr with multiple classifiers.

    This test focuses specifically on comparing different classifiers:
    - Decision tree-based (RandomForestClassifier, DecisionTreeClassifier)
    - Linear models (LogisticRegression)
    - Other algorithms (SVC, KNeighborsClassifier)
    - Comparing their performance on balanced datasets
    """
    # Step 1: Load the dataset
    output, return_code = run_command(
        f" --config-path {temp_config} load-data {test_dataset_categorical} -t target",
    )
    assert return_code == 0, "Failed to load data"

    # Step 2: Configure preprocessing
    output, return_code = run_command(
        f" --config-path {temp_config} preprocess --scale standard --handle-missing mean "
        "--categorical-features categorical_feature1 categorical_feature2 "
    )
    assert return_code == 0, "Failed to configure preprocessing"

    # Step 3: Select a single balancing technique for simplicity
    output, return_code = run_command(
        f" --config-path {temp_config} select-techniques SMOTE",
    )
    assert return_code == 0, "Failed to select technique"

    # Step 4: Select multiple classifiers from different families
    output, return_code = run_command(
        f" --config-path {temp_config} select-classifiers RandomForestClassifier LogisticRegression "
        "DecisionTreeClassifier SVC KNeighborsClassifier",
    )
    assert return_code == 0, "Failed to select classifiers"

    # Step 5: Configure metrics
    output, return_code = run_command(
        f" --config-path {temp_config} configure-metrics --metrics precision recall f1 roc_auc",
    )
    assert return_code == 0, "Failed to configure metrics"

    # Step 6: Enable visualisations focusing on classifier comparisons
    output, return_code = run_command(
        f" --config-path {temp_config} configure-visualisations --types metrics learning_curves --save-formats png "
        "--no-display",
    )
    assert return_code == 0, "Failed to configure visualisations"

    # Step 7: Enable cross-validation to better compare classifiers
    output, return_code = run_command(
        f" --config-path {temp_config} configure-evaluation --test-size 0.3 --cross-validation 3 --random-state 42",
    )
    assert return_code == 0, "Failed to configure evaluation"

    # Step 8: Run comparison
    output, return_code = run_command(
        f" --config-path {temp_config} run --output-dir {temp_output_dir}",
    )
    assert return_code == 0, "Failed to run comparison"

    # Verify results
    assert os.path.exists(temp_output_dir), "Output directory does not exist"

    # Check for all classifier directories
    for classifier in [
        "RandomForestClassifier",
        "LogisticRegression",
        "DecisionTreeClassifier",
        "SVC",
        "KNeighborsClassifier",
    ]:
        clf_dir = os.path.join(temp_output_dir, classifier)
        assert os.path.exists(clf_dir), f"{classifier} directory not generated"

        # Check for metrics on original test
        metrics_dir = os.path.join(clf_dir, "metrics_on_original_test")
        assert os.path.exists(
            metrics_dir
        ), f"Metrics directory for {classifier} not generated"

        # Check for cross-validation metrics
        cv_metrics_dir = os.path.join(clf_dir, "metrics_on_balanced_cv")
        assert os.path.exists(
            cv_metrics_dir
        ), f"CV metrics directory for {classifier} not generated"

        # Check for learning curves
        learning_curves = os.path.join(metrics_dir, "learning_curves.png")
        assert os.path.exists(
            learning_curves
        ), f"Learning curves for {classifier} not generated"

    # Verify metrics content by comparing classifiers
    # We'll use standard metrics for this comparison
    metrics_by_classifier = {}
    for classifier in [
        "RandomForestClassifier",
        "LogisticRegression",
        "DecisionTreeClassifier",
        "SVC",
        "KNeighborsClassifier",
    ]:
        results_file = os.path.join(
            temp_output_dir,
            classifier,
            "metrics_on_original_test",
            "comparison_results.csv",
        )
        results_df = pd.read_csv(results_file)

        # Extract F1 score for SMOTE
        metric_col = results_df.columns[0]  # Usually "Unnamed: 0"
        f1_idx = results_df[results_df[metric_col] == "f1"].index[0]
        f1_value = results_df.loc[f1_idx, "SMOTE"]

        metrics_by_classifier[classifier] = f1_value

    # We should have values for all classifiers
    assert len(metrics_by_classifier) == 5, "Missing metrics for some classifiers"

    # All F1 scores should be in valid range
    for classifier, f1_value in metrics_by_classifier.items():
        if not pd.isna(f1_value):
            assert 0 <= f1_value <= 1, f"F1 score for {classifier} out of valid range"
