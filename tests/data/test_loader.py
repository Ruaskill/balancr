import pytest
import numpy as np
import pandas as pd
from pathlib import Path
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
