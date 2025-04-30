import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from balancr.data import DataLoader


@pytest.fixture
def sample_csv_data():
    """Create a temporary CSV file with sample data"""
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": [0.1, 0.2, 0.3, 0.4],
            "target": ["A", "B", "A", "B"],
        }
    )

    # Save to a temporary CSV file
    temp_path = Path("temp_test.csv")
    data.to_csv(temp_path, index=False)

    yield temp_path

    # Cleanup after test
    temp_path.unlink()


@pytest.fixture
def sample_excel_data():
    """Create a temporary Excel file with sample data"""
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": [0.1, 0.2, 0.3, 0.4],
            "target": ["A", "B", "A", "B"],
        }
    )

    # Save to a temporary Excel file
    temp_path = Path("temp_test.xlsx")
    data.to_excel(temp_path, index=False)

    yield temp_path

    # Cleanup after test
    temp_path.unlink()


@pytest.fixture
def sample_csv_with_missing_targets():
    """Create a temporary CSV file with sample data containing missing target values"""
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6],
            "feature2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "target": ["A", "B", None, "A", np.nan, "B"],
        }
    )

    # Save to a temporary CSV file
    temp_path = Path("temp_missing_targets.csv")
    data.to_csv(temp_path, index=False)

    yield temp_path

    # Cleanup after test
    temp_path.unlink()


@pytest.fixture
def sample_csv_with_many_missing_targets():
    """Create a temporary CSV file with many missing target values (>20 for logging test)"""
    n_samples = 50
    data = pd.DataFrame(
        {
            "feature1": np.arange(n_samples),
            "feature2": np.random.random(n_samples),
            "target": np.array(["A"] * n_samples),
        }
    )

    # Set 25 target values to NaN (50% of the data)
    missing_indices = np.random.choice(n_samples, 25, replace=False)
    data.loc[missing_indices, "target"] = np.nan

    # Save to a temporary CSV file
    temp_path = Path("temp_many_missing_targets.csv")
    data.to_csv(temp_path, index=False)

    yield temp_path

    # Cleanup after test
    temp_path.unlink()


def test_load_data_csv_all_features(sample_csv_data):
    """Test loading CSV data with all features"""
    X, y = DataLoader.load_data(sample_csv_data, target_column="target")

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (4, 2)  # 4 samples, 2 features
    assert y.shape == (4,)  # 4 samples
    assert np.array_equal(X[:, 0], [1, 2, 3, 4])  # feature1
    assert np.array_equal(y, ["A", "B", "A", "B"])


def test_load_data_csv_selected_features(sample_csv_data):
    """Test loading CSV data with specific feature columns"""
    X, y = DataLoader.load_data(
        sample_csv_data, target_column="target", feature_columns=["feature1"]
    )

    assert X.shape == (4, 1)  # 4 samples, 1 feature
    assert np.array_equal(X.flatten(), [1, 2, 3, 4])


def test_load_data_excel(sample_excel_data):
    """Test loading Excel data"""
    X, y = DataLoader.load_data(sample_excel_data, target_column="target")

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (4, 2)
    assert y.shape == (4,)


def test_invalid_target_column(sample_csv_data):
    """Test handling of invalid target column"""
    with pytest.raises(
        ValueError, match="Target column 'invalid_target' not found in data"
    ):
        DataLoader.load_data(sample_csv_data, target_column="invalid_target")


def test_load_data_with_missing_targets(sample_csv_with_missing_targets, caplog):
    """Test loading data with missing target values"""
    caplog.set_level(logging.WARNING)

    X, y = DataLoader.load_data(sample_csv_with_missing_targets, target_column="target")

    # Check that appropriate warnings were logged
    assert "missing target values" in caplog.text.lower()

    # Check that rows with missing targets were removed
    assert X.shape[0] == 4  # Original had 6 rows, 2 with missing targets
    assert y.shape[0] == 4

    # Verify only non-missing values remain
    assert all(val is not None and not pd.isna(val) for val in y)

    # Verify the correct values were kept
    expected_y = np.array(["A", "B", "A", "B"])
    assert np.array_equal(y, expected_y)

    # Verify the correct feature values were kept
    expected_feature1 = np.array([1, 2, 4, 6])
    assert np.array_equal(X[:, 0], expected_feature1)


def test_load_data_with_many_missing_targets(
    sample_csv_with_many_missing_targets, caplog
):
    """Test loading data with many missing target values (tests summary logging)"""
    caplog.set_level(logging.WARNING)

    X, y = DataLoader.load_data(
        sample_csv_with_many_missing_targets, target_column="target"
    )

    # Check that appropriate warnings were logged
    assert "missing target values" in caplog.text.lower()

    # Should include percentage warning since >10% were removed
    assert "significant portion" in caplog.text.lower()

    # For many missing values, we should see a summary rather than full list
    assert "first few row indices" in caplog.text.lower()

    # Check that rows with missing targets were removed
    assert X.shape[0] == 25  # Original had 50 rows, 25 with missing targets
    assert y.shape[0] == 25

    # All remaining values should be "A" (as per fixture)
    assert all(val == "A" for val in y)


def test_load_data_no_missing_targets(sample_csv_data, caplog):
    """Test loading data with no missing target values"""
    caplog.set_level(logging.WARNING)

    X, y = DataLoader.load_data(sample_csv_data, target_column="target")

    # No warnings about missing target values should be logged
    assert "missing target values" not in caplog.text.lower()

    # Data should be loaded as is
    assert X.shape[0] == 4
    assert y.shape[0] == 4


def test_invalid_feature_columns(sample_csv_data):
    """Test handling of invalid feature columns"""
    with pytest.raises(ValueError, match="Feature columns not found:"):
        DataLoader.load_data(
            sample_csv_data, target_column="target", feature_columns=["invalid_feature"]
        )


def test_unsupported_file_format():
    """Test handling of unsupported file format"""
    invalid_path = Path("test.txt")
    with pytest.raises(ValueError, match="Unsupported file format:"):
        DataLoader.load_data(invalid_path, target_column="target")
