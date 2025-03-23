import pytest
import numpy as np
from data.preprocessor import DataPreprocessor


@pytest.fixture
def preprocessor():
    """Create a preprocessor instance"""
    return DataPreprocessor()


@pytest.fixture
def sample_data():
    """Create sample data with known properties"""
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, np.nan, 12.0],  # Include a missing value
        ]
    )
    y = np.array(["A", "B", "A", "B"])
    return X, y


@pytest.fixture
def correlated_data():
    """Create sample data with highly correlated features"""
    # Create perfectly correlated features
    feature1 = np.array([1, 2, 3, 4])
    feature2 = feature1 * 2  # Perfectly correlated with feature1
    feature3 = np.array([7, 8, 9, 10])  # Independent feature

    X = np.column_stack([feature1, feature2, feature3])
    return X


def test_inspect_class_distribution(preprocessor, sample_data):
    """Test class distribution inspection"""
    _, y = sample_data
    distribution = preprocessor.inspect_class_distribution(y)

    assert isinstance(distribution, dict)
    assert len(distribution) == 2
    assert distribution["A"] == 2
    assert distribution["B"] == 2


def test_check_data_quality_missing_values(preprocessor, sample_data):
    """Test detection of missing values"""
    X, _ = sample_data
    quality_report = preprocessor.check_data_quality(X)

    assert "missing_values" in quality_report
    assert isinstance(quality_report["missing_values"], np.ndarray)
    assert (
        quality_report["missing_values"][1] == 1
    )  # One missing value in second column


def test_check_data_quality_constant_features(preprocessor):
    """Test detection of constant features"""
    X = np.array([[1, 5, 3], [1, 5, 4], [1, 5, 5]])
    quality_report = preprocessor.check_data_quality(X)

    assert "constant_features" in quality_report
    assert (
        len(quality_report["constant_features"]) == 2
    )  # First two columns are constant


def test_check_data_quality_correlations(preprocessor, correlated_data):
    """Test detection of highly correlated features"""
    quality_report = preprocessor.check_data_quality(
        correlated_data, feature_names=["feature1", "feature2", "feature3"]
    )

    assert "feature_correlations" in quality_report
    correlations = quality_report["feature_correlations"]
    assert len(correlations) > 0
    # Check if the perfectly correlated features are detected
    assert any(corr[2] > 0.95 for corr in correlations)


def test_preprocess_scaling(preprocessor, sample_data):
    """Test feature scaling"""
    X, y = sample_data
    X_processed, y_processed = preprocessor.preprocess(
        X, y, scale=True, encode=False
    )

    # Check if data is scaled (mean ≈ 0, std ≈ 1)
    assert np.abs(X_processed.mean()) < 1e-10
    assert np.abs(X_processed.std() - 1) < 1e-10


def test_preprocess_label_encoding(preprocessor, sample_data):
    """Test label encoding"""
    X, y = sample_data
    X_processed, y_processed = preprocessor.preprocess(
        X, y, scale=False, encode="auto"
    )

    assert y_processed.dtype == np.int64
    assert set(y_processed) == {0, 1}  # Binary encoded labels


def test_preprocess_missing_values(preprocessor):
    """Test handling of missing values"""
    X = np.array([[1, np.nan, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array(["A", "B", "A"])

    X_processed, _ = preprocessor.preprocess(X, y)

    # Check if missing values are replaced
    assert not np.isnan(X_processed).any()


def test_preprocess_no_modifications(preprocessor, sample_data):
    """Test preprocessing with no modifications"""
    X, y = sample_data
    X_processed, y_processed = preprocessor.preprocess(
        X, y, scale=False, encode=False
    )

    # Only missing values should be handled
    assert not np.array_equal(X, X_processed)  # Different due to missing value handling
    assert np.array_equal(y, y_processed)  # Labels should remain unchanged
