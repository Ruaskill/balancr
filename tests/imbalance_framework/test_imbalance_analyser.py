import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from imbalance_framework.imbalance_analyser import BalancingFramework
from imbalance_framework.base import BaseBalancer


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
def mock_registry(monkeypatch):
    """Create a mock registry with test techniques"""
    mock = MagicMock()
    mock.list_available_techniques.return_value = {
        "custom": ["CustomTechnique"],
        "imblearn": ["SMOTE", "RandomUnderSampler"],
    }
    mock.get_technique_class.return_value = MockTechnique
    monkeypatch.setattr(
        "imbalance_framework.technique_registry.TechniqueRegistry", lambda: mock
    )
    return mock


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
    assert abs(framework.X.mean()) < 1e-10


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

    results = framework.apply_balancing_techniques(technique_names=["SMOTE", "RandomUnderSampler"])

    assert isinstance(results, dict)
    assert len(results) == 2
    assert "SMOTE" in results
    assert "RandomUnderSampler" in results


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
    framework.generate_learning_curves(classifier_name="RandomForestClassifier", save_path=learning_curve_path)
    assert learning_curve_path.exists()


def test_generate_learning_curves_no_data(framework):
    """Test learning curve generation without data"""
    with pytest.raises(ValueError):
        framework.generate_learning_curves(classifier_name="RandomForestClassifier")


def test_handle_quality_issues(framework):
    """Test handling of data quality issues"""
    quality_report = {
        "missing_values": np.array([True, False]),
        "constant_features": np.array([0]),
        "feature_correlations": [("feature1", "feature2", 0.96)],
    }

    # Should not raise any exceptions
    framework._handle_quality_issues(quality_report)
