import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from balancr import BalancingFramework, BaseBalancer


# Mock classes and fixtures
class MockTechnique(BaseBalancer):
    def balance(self, X, y):
        # Return the same data for testing
        return X, y


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])
    return X, y


@pytest.fixture
def sample_dataframe(sample_data):
    """Create a sample DataFrame"""
    X, y = sample_data
    df = pd.DataFrame(X, columns=["feature_1", "feature_2"])
    df["target"] = y
    return df


@pytest.fixture
def framework():
    """Create a new framework instance for each test"""
    return BalancingFramework()


@pytest.fixture
def mock_classifier():
    """Create a mock classifier that returns predictable results"""
    mock_clf = MagicMock()
    mock_clf.predict.return_value = np.array([0, 1, 0, 1])
    mock_clf.predict_proba.return_value = np.array(
        [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8]]
    )

    return mock_clf


@pytest.fixture
def framework_with_balanced_data(mock_classifier):
    """Create a framework instance with mock balanced datasets"""
    framework = BalancingFramework()

    # Test data
    framework.X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    framework.y_test = np.array([0, 1, 0, 1])

    # Mock balanced datasets
    framework.current_balanced_datasets = {
        "SMOTE": {
            "X_balanced": np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
            "y_balanced": np.array([0, 1, 0, 1]),
        },
        "RandomUnderSampler": {
            "X_balanced": np.array([[1, 2], [7, 8]]),
            "y_balanced": np.array([0, 1]),
        },
    }

    # Mock classifier registry
    framework.classifier_registry = MagicMock()

    # Mock classifier class that returns our mock classifier
    mock_classifier_class = MagicMock()
    mock_classifier_class.return_value = mock_classifier

    framework.classifier_registry.get_classifier_class.return_value = (
        mock_classifier_class
    )

    return framework


@pytest.fixture
def mock_registry(monkeypatch):
    """Create a mock registry with test techniques"""
    mock = MagicMock()
    mock.list_available_techniques.return_value = {
        "custom": ["CustomTechnique"],
        "imblearn": ["SMOTE", "RandomUnderSampler"],
    }
    mock.get_technique_class.return_value = MockTechnique
    monkeypatch.setattr(
        "balancr.technique_registry.TechniqueRegistry", lambda: mock
    )
    return mock


def test_train_classifiers_no_balanced_data():
    """Test train_classifiers with no balanced datasets"""
    framework = BalancingFramework()

    with pytest.raises(ValueError, match="No balanced datasets available"):
        framework.train_classifiers()


def test_train_classifiers_no_test_data(framework_with_balanced_data):
    """Test train_classifiers with no test data"""
    framework = framework_with_balanced_data
    framework.X_test = None
    framework.y_test = None

    with pytest.raises(ValueError, match="Test data not found"):
        framework.train_classifiers()


@patch("balancr.imbalance_analyser.get_metrics")
def test_train_classifiers_default_classifier(
    mock_get_metrics, framework_with_balanced_data
):
    """Test train_classifiers with default classifier"""
    framework = framework_with_balanced_data

    # Mock the get_metrics function
    mock_get_metrics.return_value = {
        "accuracy": 0.75,
        "precision": 0.8,
        "recall": 0.7,
        "f1": 0.75,
    }

    # Call the function with default parameters
    results = framework.train_classifiers()

    # Check that results have the expected structure
    assert "RandomForestClassifier" in results
    assert "SMOTE" in results["RandomForestClassifier"]
    assert "RandomUnderSampler" in results["RandomForestClassifier"]
    assert "standard_metrics" in results["RandomForestClassifier"]["SMOTE"]

    # Check that the classifier registry was called correctly
    framework.classifier_registry.get_classifier_class.assert_called_with(
        "RandomForestClassifier"
    )

    # Check that get_metrics was called for each technique
    assert mock_get_metrics.call_count == 2


@patch("balancr.imbalance_analyser.get_metrics")
@patch("balancr.imbalance_analyser.get_cv_scores")
def test_train_classifiers_with_cv(
    mock_get_cv_scores, mock_get_metrics, framework_with_balanced_data
):
    """Test train_classifiers with cross-validation enabled"""
    framework = framework_with_balanced_data

    # Mock metric functions
    mock_get_metrics.return_value = {
        "accuracy": 0.75,
        "precision": 0.8,
        "recall": 0.7,
        "f1": 0.75,
    }

    mock_get_cv_scores.return_value = {
        "cv_accuracy_mean": 0.73,
        "cv_accuracy_std": 0.05,
        "cv_precision_mean": 0.78,
        "cv_precision_std": 0.07,
    }

    # Call function with CV enabled
    results = framework.train_classifiers(enable_cv=True, cv_folds=3)

    # Check that CV metrics are included
    assert "cv_metrics" in results["RandomForestClassifier"]["SMOTE"]
    assert "cv_metrics" in results["RandomForestClassifier"]["RandomUnderSampler"]

    # Check that get_cv_scores was called with the right parameters
    mock_get_cv_scores.assert_called_with(
        framework.classifier_registry.get_classifier_class().return_value,
        framework.current_balanced_datasets["RandomUnderSampler"]["X_balanced"],
        framework.current_balanced_datasets["RandomUnderSampler"]["y_balanced"],
        n_folds=3,
    )


@patch("balancr.imbalance_analyser.get_metrics")
def test_train_classifiers_custom_classifiers(
    mock_get_metrics, framework_with_balanced_data
):
    """Test train_classifiers with custom classifier configurations"""
    framework = framework_with_balanced_data

    # Create two separate mock classifiers
    mock_rf = MagicMock()
    mock_lr = MagicMock()

    # Update the side_effect to use a function that will return different mocks
    def get_classifier_by_name(name):
        if name == "RandomForestClassifier":
            return lambda **kwargs: mock_rf
        elif name == "LogisticRegression":
            return lambda **kwargs: mock_lr
        else:
            return None

    framework.classifier_registry.get_classifier_class.side_effect = (
        get_classifier_by_name
    )

    # Mock get_metrics
    mock_get_metrics.return_value = {
        "accuracy": 0.75,
        "precision": 0.8,
        "recall": 0.7,
        "f1": 0.75,
    }

    # Define custom classifiers
    classifier_configs = {
        "RandomForestClassifier": {"n_estimators": 100, "random_state": 42},
        "LogisticRegression": {"C": 1.0, "random_state": 42},
        "NonExistentClassifier": {"param": "value"},
    }

    # Call the function with custom classifiers
    results = framework.train_classifiers(classifier_configs=classifier_configs)

    assert "RandomForestClassifier" in results
    assert "LogisticRegression" in results
    assert "NonExistentClassifier" not in results

    # Verify that both mocks were used
    assert mock_rf.fit.called
    assert mock_lr.fit.called

    # Check that both SMOTE and RandomUnderSampler were processed for each classifier
    techniques = results["RandomForestClassifier"].keys()
    assert "SMOTE" in techniques
    assert "RandomUnderSampler" in techniques

    techniques = results["LogisticRegression"].keys()
    assert "SMOTE" in techniques
    assert "RandomUnderSampler" in techniques


@patch("balancr.imbalance_analyser.get_metrics")
@patch("logging.error")
def test_train_classifiers_with_error(
    mock_logging_error, mock_get_metrics, framework_with_balanced_data
):
    """Test train_classifiers handling of exceptions during training"""
    framework = framework_with_balanced_data

    # Make the classifier raise an exception for one technique
    framework.classifier_registry.get_classifier_class().return_value.fit.side_effect = [
        Exception("Training error"),
        None,
    ]

    # Mock get_metrics for the successful technique
    mock_get_metrics.return_value = {
        "accuracy": 0.75,
        "precision": 0.8,
        "recall": 0.7,
        "f1": 0.75,
    }

    results = framework.train_classifiers()

    # Check that only the successful technique is in the results
    assert "RandomForestClassifier" in results
    assert len(results["RandomForestClassifier"]) == 1
    assert "RandomUnderSampler" in results["RandomForestClassifier"]
    assert "SMOTE" not in results["RandomForestClassifier"]

    # Check that the error was logged
    mock_logging_error.assert_called_once()
    assert "Training error" in mock_logging_error.call_args[0][0]


def test_train_classifiers_no_classifiers_found(framework_with_balanced_data):
    """Test train_classifiers when no valid classifiers are found"""
    framework = framework_with_balanced_data

    # Make the classifier registry return None for all classifiers
    framework.classifier_registry.get_classifier_class.return_value = None

    results = framework.train_classifiers()

    assert results == {}


def test_framework_initialization(framework):
    """Test framework initialization"""
    assert framework.X is None
    assert framework.y is None
    assert isinstance(framework.results, dict)
    assert isinstance(framework.current_data_info, dict)
    assert isinstance(framework.current_balanced_datasets, dict)


@patch("pandas.read_csv")
def test_load_data_csv(mock_read_csv, framework, sample_dataframe):
    """Test loading data from CSV"""
    mock_read_csv.return_value = sample_dataframe

    framework.load_data(
        file_path="test.csv",
        target_column="target",
        feature_columns=["feature_1", "feature_2"],
    )

    assert framework.X is not None
    assert framework.y is not None
    assert framework.current_data_info["file_path"] == "test.csv"
    assert framework.current_data_info["target_column"] == "target"


@patch("pandas.read_excel")
def test_load_data_excel(mock_read_excel, framework, sample_dataframe):
    """Test loading data from Excel"""
    mock_read_excel.return_value = sample_dataframe

    framework.load_data(
        file_path="test.xlsx",
        target_column="target",
        feature_columns=["feature_1", "feature_2"],
    )

    assert framework.X is not None
    assert framework.y is not None
    assert framework.current_data_info["file_path"] == "test.xlsx"
    assert framework.current_data_info["target_column"] == "target"


def test_load_data_invalid_file(framework):
    """Test loading data from unsupported file format"""
    with pytest.raises(ValueError):
        framework.load_data(file_path="test.txt", target_column="target")


def test_inspect_class_distribution(framework, sample_data):
    """Test class distribution inspection"""
    X, y = sample_data
    framework.X = X
    framework.y = y

    distribution = framework.inspect_class_distribution(display=False)
    assert isinstance(distribution, dict)
    assert set(distribution.keys()) == {0, 1}
    assert distribution[0] == 2  # Count of class 0
    assert distribution[1] == 2  # Count of class 1


def test_preprocess_data_no_data(framework):
    """Test preprocessing without loading data first"""
    with pytest.raises(ValueError):
        framework.preprocess_data()


def test_preprocess_data(framework, sample_data):
    """Test data preprocessing"""
    X, y = sample_data
    framework.X = X
    framework.y = y

    framework.preprocess_data()
    assert framework.X is not None
    assert framework.y is not None
    # Check if data was scaled (mean should be close to 0)
    assert (framework.X.mean().abs() < 1e-10).all()


def test_compare_techniques_no_data(framework):
    """Test comparing techniques without loading data"""
    with pytest.raises(ValueError):
        framework.apply_balancing_techniques(["SMOTE"])


def test_compare_techniques(framework):
    """Test technique comparison with sufficient data for SMOTE"""
    # Create larger dataset that works with SMOTE. Could not mock behaviour properly
    np.random.seed(42)
    X = np.random.rand(20, 5)
    y = np.concatenate([np.zeros(10), np.ones(10)])

    framework.X = X
    framework.y = y

    results = framework.apply_balancing_techniques(
        technique_names=["SMOTE", "RandomUnderSampler"]
    )

    assert isinstance(results, dict)
    assert len(results) == 2
    assert "SMOTE" in results
    assert "RandomUnderSampler" in results


def test_apply_balancing_techniques_invalid_technique(framework_with_balanced_data):
    """Test apply_balancing_techniques with an invalid technique name"""
    framework = framework_with_balanced_data

    # Test data
    framework.X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    framework.y = np.array([0, 1, 0, 1])

    # Configure technique registry to return None for an invalid technique
    framework.technique_registry = MagicMock()
    framework.technique_registry.get_technique_class.return_value = None
    framework.list_available_techniques = MagicMock(
        return_value={"custom": [], "imblearn": ["SMOTE", "RandomUnderSampler"]}
    )

    # Test with invalid technique name
    with pytest.raises(ValueError) as excinfo:
        framework.apply_balancing_techniques(["InvalidTechnique"])

    # Verify the error message
    error_message = str(excinfo.value)
    assert "Technique 'InvalidTechnique' not found" in error_message
    assert "Available techniques" in error_message

    # Verify list_available_techniques was called
    framework.list_available_techniques.assert_called_once()


def test_save_results_no_results(framework):
    """Test saving results without running comparison first"""
    with pytest.raises(ValueError):
        framework.save_results("results.csv")


def test_save_results(framework, tmp_path):
    """Test saving results to file"""
    framework.results = {
        "SMOTE": {
            "accuracy": 0.8,
            "precision": 0.7,
            "recall": 0.9,
            "f1": 0.8,
            "roc_auc": 0.85,
        },
        "RandomUnderSampler": {
            "accuracy": 0.75,
            "precision": 0.8,
            "recall": 0.6,
            "f1": 0.7,
            "roc_auc": 0.82,
        },
    }

    # Test CSV saving
    csv_path = tmp_path / "results.csv"
    framework.save_results(csv_path, file_type="csv", include_plots=False)
    assert csv_path.exists()

    # Test JSON saving
    json_path = tmp_path / "results.json"
    framework.save_results(json_path, file_type="json", include_plots=False)
    assert json_path.exists()


def test_save_results_invalid_type(framework):
    """Test saving results with invalid file type"""
    framework.results = {"test": {"metric": 0.5}}
    with pytest.raises(ValueError):
        framework.save_results("results.txt", file_type="txt")


def test_generate_balanced_data_no_data(framework):
    """Test generating balanced data without running comparison first"""
    with pytest.raises(ValueError):
        framework.generate_balanced_data("output/")


def test_generate_balanced_data(framework, sample_data, tmp_path):
    """Test generating balanced datasets"""
    X, y = sample_data
    framework.X = X
    framework.y = y
    framework.current_data_info = {
        "feature_columns": ["feature_1", "feature_2"],
        "target_column": "target",
    }
    framework.current_balanced_datasets = {"SMOTE": {"X_balanced": X, "y_balanced": y}}

    output_dir = tmp_path / "balanced"
    framework.generate_balanced_data(str(output_dir))

    assert (output_dir / "balanced_SMOTE.csv").exists()


def test_compare_balanced_class_distributions(framework, sample_data, tmp_path):
    """Test comparing class distributions"""
    X, y = sample_data
    framework.current_balanced_datasets = {
        "SMOTE": {"X_balanced": X, "y_balanced": y},
        "RandomUnderSampler": {"X_balanced": X, "y_balanced": y},
    }

    balanced_class_path = tmp_path / "balanced_class_distributions.png"
    framework.compare_balanced_class_distributions(save_path=balanced_class_path)
    assert balanced_class_path.exists()


def test_compare_balanced_class_distributions_no_data(framework):
    """Test comparing class distributions without data"""
    with pytest.raises(ValueError):
        framework.compare_balanced_class_distributions()


def test_generate_learning_curves(framework, tmp_path):
    """Test learning curve generation with sufficient data"""
    # Larger dataset
    np.random.seed(42)
    X = np.random.rand(20, 5)
    y = np.concatenate([np.zeros(10), np.ones(10)])

    framework.current_balanced_datasets = {"SMOTE": {"X_balanced": X, "y_balanced": y}}

    learning_curve_path = tmp_path / "learning_curve.png"
    framework.generate_learning_curves(
        classifier_name="RandomForestClassifier",
        save_path=learning_curve_path,
        learning_curve_type="Balanced Datasets"
    )
    assert learning_curve_path.exists()


def test_generate_learning_curves_no_data(framework):
    """Test learning curve generation without data"""
    with pytest.raises(ValueError):
        framework.generate_learning_curves(
            classifier_name="RandomForestClassifier",
            learning_curve_type="Balanced Datasets"
        )


def test_handle_quality_issues(framework):
    """Test handling of data quality issues"""
    quality_report = {
        "missing_values": [(0, 1)],  # row 0, column 1 missing
        "constant_features": [("feature1", 0)],
        "feature_correlations": [("feature1", "feature2", 0.96)],
    }

    # Should not raise any exceptions
    framework._handle_quality_issues(quality_report)
