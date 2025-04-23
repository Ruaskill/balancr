import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from balancr.data import DataPreprocessor
from sklearn.preprocessing import MinMaxScaler, RobustScaler


@pytest.fixture
def preprocessor():
    """Create a preprocessor instance"""
    return DataPreprocessor()


@pytest.fixture
def sample_data():
    """Create sample data with known properties"""
    X = np.array(
        [
            [1.0, 2.0, 3.0, "cat1"],
            [4.0, 5.0, 6.0, "cat2"],
            [7.0, 8.0, 9.0, "cat1"],
            [10.0, np.nan, 12.0, "cat3"],  # Include a missing value
        ],
        dtype=object,
    )
    X_df = pd.DataFrame(X, columns=["num1", "num2", "num3", "cat1"])
    y = np.array(["A", "B", "A", "B"])
    return X_df, y


@pytest.fixture
def correlated_data():
    """Create sample data with highly correlated features"""
    # Create perfectly correlated features
    feature1 = np.array([1, 2, 3, 4])
    feature2 = feature1 * 2  # Perfectly correlated with feature1
    feature3 = np.array([7, 8, 9, 10])  # Independent feature

    X = np.column_stack([feature1, feature2, feature3])
    X_df = pd.DataFrame(X, columns=["feature1", "feature2", "feature3"])
    return X_df


@pytest.fixture
def categorical_data():
    """Create sample data with categorical features"""
    data = {
        "num_feature": [1.0, 2.0, 3.0, 4.0, 5.0],
        "low_cardinality": ["A", "B", "A", "C", "B"],
        "high_cardinality": [f"val_{i}" for i in range(5)],
        "skewed_distribution": ["dominant", "dominant", "dominant", "dominant", "rare"],
        "ordinal_feature": ["low", "medium", "high", "medium", "low"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def categorical_data_for_auto_encoding():
    """Create sample data with categorical features for auto-encoding testing"""
    # Create 30 rows of data
    n_samples = 100

    # Generate data
    data = {
        "num_feature": np.linspace(1.0, 5.0, n_samples),
        # Low cardinality - 3 categories, evenly distributed
        "low_cardinality": ["A", "B", "C", "D"] * 25,
        # High cardinality - 30 unique values
        "high_cardinality": [f"val_{i}" for i in range(n_samples)],
        # Skewed distribution - one dominant value (80%)
        "skewed_distribution": ["dominant"] * 80 + [f"rare_{i}" for i in range(20)],
        # Ordinal feature - low, medium, high values
        "ordinal_feature": ["low"] * 33 + ["medium"] * 33 + ["high"] * 34,
        # Many rare categories, under 50 unique values - 44% dominant, 56% rare
        "many_rare_low_card": ["common"] * 44 + [f"rare_{i}" for i in range(28)] * 2,
    }

    df = pd.DataFrame(data)

    return df


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
    assert isinstance(quality_report["missing_values"], list)
    # Check that the second column has a missing value
    assert any(
        item[0] == "feature_1" or item[0] == X.columns[1]
        for item in quality_report["missing_values"]
    )
    # Check that it has 1 missing value
    missing_col = next(
        item
        for item in quality_report["missing_values"]
        if item[0] == "feature_1" or item[0] == X.columns[1]
    )
    assert missing_col[1] == 1


def test_check_data_quality_constant_features(preprocessor):
    """Test detection of constant features"""
    X = pd.DataFrame(
        {"constant1": [1, 1, 1], "constant2": ["a", "a", "a"], "varying": [1, 2, 3]}
    )
    quality_report = preprocessor.check_data_quality(X)

    assert "constant_features" in quality_report
    assert isinstance(quality_report["constant_features"], list)
    # Should identify exactly two constant columns
    constant_columns = [item[0] for item in quality_report["constant_features"]]
    assert len(constant_columns) == 2
    assert "constant1" in constant_columns
    assert "constant2" in constant_columns
    assert "varying" not in constant_columns


def test_check_data_quality_correlations(preprocessor):
    """Test detection of highly correlated features"""
    # Create data with perfect correlation
    X = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],  # Perfect correlation with feature1
            "feature3": [5, 6, 7, 8, 9],  # Not perfectly correlated
        }
    )

    quality_report = preprocessor.check_data_quality(X)

    assert "feature_correlations" in quality_report
    assert isinstance(quality_report["feature_correlations"], list)
    correlations = quality_report["feature_correlations"]
    assert len(correlations) > 0

    # Find correlation between feature1 and feature2
    feature1_feature2_corr = None
    for corr in correlations:
        if (corr[0] == "feature1" and corr[1] == "feature2") or (
            corr[0] == "feature2" and corr[1] == "feature1"
        ):
            feature1_feature2_corr = corr
            break

    assert feature1_feature2_corr is not None
    assert feature1_feature2_corr[2] > 0.95


def test_check_data_quality_no_issues(preprocessor):
    """Test when there are no quality issues"""
    # Create clean data with no issues
    X = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": [8, 6, 7, 5],  # Not correlated with feature1
        }
    )

    quality_report = preprocessor.check_data_quality(X)

    assert quality_report["missing_values"] == []
    assert quality_report["constant_features"] == []
    assert quality_report["feature_correlations"] == []


def test_check_data_quality_with_numpy_input(preprocessor):
    """Test handling of numpy array input"""
    # Create a numpy array
    X = np.array([[1, 5], [2, 6], [3, 7], [4, 8]])

    quality_report = preprocessor.check_data_quality(X)

    assert isinstance(quality_report["missing_values"], list)
    assert isinstance(quality_report["constant_features"], list)
    assert isinstance(quality_report["feature_correlations"], list)


def test_check_data_quality_with_mixed_types(preprocessor):
    """Test handling of mixed data types"""
    # Create data with mixed types
    X = pd.DataFrame(
        {
            "numeric": [1, 2, 3, 4],
            "string": ["a", "b", "c", "d"],
            "boolean": [True, False, True, False],
            "dates": pd.date_range("2023-01-01", periods=4),
        }
    )

    quality_report = preprocessor.check_data_quality(X)

    # Should handle mixed types without errors
    assert isinstance(quality_report["missing_values"], list)
    assert isinstance(quality_report["constant_features"], list)
    # Should only calculate correlations for numeric columns
    assert isinstance(quality_report["feature_correlations"], list)


def test_preprocess_basic(preprocessor, sample_data):
    """Test basic preprocessing with default parameters"""
    X, y = sample_data
    X_processed, y_processed = preprocessor.preprocess(
        X,
        y,
        categorical_features={"cat1": "label"},
        all_features=["num1", "num2", "num3", "cat1"],
        # scale="none"
    )

    # Check output types
    assert isinstance(X_processed, pd.DataFrame)
    assert isinstance(y_processed, np.ndarray)

    # Check that missing values were handled
    assert not X_processed.isna().any().any()

    # Check that categorical column was transformed
    assert "cat1" in X_processed.columns
    assert X_processed["cat1"].dtype == np.int64


def test_preprocess_with_scaling(preprocessor, sample_data):
    """Test feature scaling"""
    X, y = sample_data
    X_processed, _ = preprocessor.preprocess(
        X,
        y,
        scale="standard",
        categorical_features={"cat1": "label"},
        all_features=["num1", "num2", "num3", "cat1"],
    )

    # Check if numerical columns were scaled (mean ≈ 0, std ≈ 1)
    numerical_cols = ["num1", "num2", "num3"]
    for col in numerical_cols:
        if col in X_processed.columns:
            assert abs(X_processed[col].mean()) < 1e-10
            assert abs(X_processed[col].std(ddof=0) - 1) < 1e-10


def test_preprocess_minmax_scaling(preprocessor, sample_data):
    """Test that minmax scaling works correctly"""
    X_df, y = sample_data

    # Convert categorical column to string to avoid warnings
    X_df["cat1"] = X_df["cat1"].astype(str)

    # Only scale numerical columns
    numerical_cols = ["num1", "num2", "num3"]
    categorical_cols = {"cat1": "none"}

    # Apply processing with minmax scaling
    X_processed, y_processed = preprocessor.preprocess(
        X_df,
        y,
        scale="minmax",
        handle_missing="mean",
        categorical_features=categorical_cols,
    )

    # Manually apply MinMaxScaler to verify
    manual_scaler = MinMaxScaler()
    # Fill missing values first to match the preprocess function
    X_df_numerical = X_df[numerical_cols].copy()
    X_df_numerical = X_df_numerical.fillna(X_df_numerical.mean())
    X_df_scaled_expected = manual_scaler.fit_transform(X_df_numerical)

    # Extract scaled numerical columns from processed data
    X_processed_numerical = X_processed[numerical_cols].values

    # Check that values are between 0 and 1
    assert np.all((X_processed_numerical >= 0) & (X_processed_numerical <= 1))

    # The values should match the manual scaling
    np.testing.assert_allclose(X_processed_numerical, X_df_scaled_expected)

    # Categorical column should remain unchanged
    assert "cat1" in X_processed.columns

    # Verify that the preprocessor set the feature names correctly
    assert set(preprocessor.feature_names) == set(["num1", "num2", "num3", "cat1"])


def test_preprocess_robust_scaling(preprocessor, sample_data):
    """Test that robust scaling works correctly"""
    X_df, y = sample_data

    # Convert categorical column to string to avoid warnings
    X_df["cat1"] = X_df["cat1"].astype(str)

    # Only scale numerical columns
    numerical_cols = ["num1", "num2", "num3"]
    categorical_cols = {"cat1": "none"}

    # Apply processing with robust scaling
    X_processed, y_processed = preprocessor.preprocess(
        X_df,
        y,
        scale="robust",
        handle_missing="mean",
        categorical_features=categorical_cols,
    )

    # Manually apply RobustScaler to verify
    manual_scaler = RobustScaler()
    # Fill missing values first to match the preprocess function
    X_df_numerical = X_df[numerical_cols].copy()
    X_df_numerical = X_df_numerical.fillna(X_df_numerical.mean())
    X_df_scaled_expected = manual_scaler.fit_transform(X_df_numerical)

    # Extract scaled numerical columns from processed data
    X_processed_numerical = X_processed[numerical_cols].values

    # Check that the median of each feature is approx zero
    feature_medians = np.median(X_processed_numerical, axis=0)
    assert np.allclose(feature_medians, 0, atol=1e-10)

    # The values should match the manual scaling
    np.testing.assert_allclose(X_processed_numerical, X_df_scaled_expected)

    # Categorical column should remain unchanged
    assert "cat1" in X_processed.columns

    # Verify that the preprocessor set the feature names correctly
    assert set(preprocessor.feature_names) == set(["num1", "num2", "num3", "cat1"])


def test_preprocess_with_different_scalers(preprocessor):
    """Test all scaling options on a more controlled dataset"""
    # Create a simple dataset with outliers to better test the differences between scalers
    X = pd.DataFrame(
        {
            "feature1": [0, 1, 2, 3, 100],  # With outlier
            "feature2": [5, 6, 7, 8, 9],  # No outliers
        }
    )
    y = np.array([0, 1, 0, 1, 0])

    # Test standard scaling
    X_standard, _ = preprocessor.preprocess(X.copy(), y, scale="standard")

    # Test minmax scaling
    X_minmax, _ = preprocessor.preprocess(X.copy(), y, scale="minmax")

    # Test robust scaling
    X_robust, _ = preprocessor.preprocess(X.copy(), y, scale="robust")

    # MinMax scaling should result in values between 0 and 1
    assert np.all((X_minmax.values >= 0) & (X_minmax.values <= 1))

    # For RobustScaler, check that the median is approximately zero
    assert abs(np.median(X_robust["feature1"])) < 1e-10
    assert abs(np.median(X_robust["feature2"])) < 1e-10

    # For StandardScaler, check that the mean is approximately zero
    # and standard deviation is approximately one
    assert abs(X_standard["feature1"].mean()) < 1e-10
    assert abs(X_standard["feature2"].mean()) < 1e-10
    assert abs(X_standard["feature1"].std(ddof=0) - 1.0) < 1e-10
    assert abs(X_standard["feature2"].std(ddof=0) - 1.0) < 1e-10

    # For MinMaxScaler, check that the minimum is 0 and maximum is 1
    assert abs(X_minmax["feature1"].min()) < 1e-10
    assert abs(X_minmax["feature2"].min()) < 1e-10
    assert abs(X_minmax["feature1"].max() - 1.0) < 1e-10
    assert abs(X_minmax["feature2"].max() - 1.0) < 1e-10


def test_handle_constant_features_drop(preprocessor):
    """Test dropping constant features."""
    # Create test data with a constant feature
    data = pd.DataFrame(
        {
            "constant_feature": [5, 5, 5, 5],
            "normal_feature": [1, 2, 3, 4],
            "another_normal": [10, 20, 30, 40],
        }
    )
    y = np.array([0, 1, 0, 1])

    constant_features = ["constant_feature"]

    # Preprocess with dropping constant features
    X_processed, y_processed = preprocessor.preprocess(
        data,
        y,
        handle_constant_features="drop",
        constant_features=constant_features,
        all_features=data.columns.tolist(),
    )

    # Check that constant feature was dropped
    assert "constant_feature" not in X_processed.columns
    assert list(X_processed.columns) == ["normal_feature", "another_normal"]


def test_handle_constant_features_none(preprocessor):
    """Test not handling constant features when option is 'none'."""
    # Create test data with a constant feature
    data = pd.DataFrame(
        {"constant_feature": [5, 5, 5, 5], "normal_feature": [1, 2, 3, 4]}
    )
    y = np.array([0, 1, 0, 1])

    constant_features = ["constant_feature"]

    # Preprocess without handling constant features
    X_processed, y_processed = preprocessor.preprocess(
        data,
        y,
        handle_constant_features="none",
        constant_features=constant_features,
        all_features=data.columns.tolist(),
    )

    # Check that constant feature was retained
    assert "constant_feature" in X_processed.columns
    assert list(X_processed.columns) == ["constant_feature", "normal_feature"]


def test_handle_correlations_drop_first(preprocessor):
    """Test dropping first feature in correlated pairs."""
    # Create test data with two correlated features
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": [2, 4, 6, 8],  # feature2 = 2 * feature1
            "feature3": [10, 20, 30, 40],
        }
    )
    y = np.array([0, 1, 0, 1])

    correlated_features = [("feature1", "feature2", 1.0)]

    # Preprocess with drop_first option
    X_processed, y_processed = preprocessor.preprocess(
        data,
        y,
        handle_correlations="drop_first",
        correlated_features=correlated_features,
        all_features=data.columns.tolist(),
    )

    # Check that first feature was dropped
    assert "feature1" not in X_processed.columns
    assert "feature2" in X_processed.columns
    assert list(X_processed.columns) == ["feature2", "feature3"]


def test_handle_correlations_drop_lowest(preprocessor):
    """Test dropping feature with lowest variance in correlated pairs."""
    # Create test data with two correlated features, one with lower variance
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],  # Variance: 1.25
            "feature2": [10, 20, 30, 45],  # Variance: 187.5 (much higher)
            "feature3": [5, 6, 7, 8],
        }
    )
    y = np.array([0, 1, 0, 1])

    correlated_features = [("feature1", "feature2", 0.98)]

    # Preprocess with drop_lowest option
    X_processed, y_processed = preprocessor.preprocess(
        data,
        y,
        handle_correlations="drop_lowest",
        correlated_features=correlated_features,
        all_features=data.columns.tolist(),
    )

    # Check that feature with lowest variance was dropped
    assert "feature1" not in X_processed.columns  # Should be dropped (lower variance)
    assert "feature2" in X_processed.columns
    assert list(X_processed.columns) == ["feature2", "feature3"]


def test_handle_correlations_pca(preprocessor):
    """Test handling correlations with PCA."""
    # Create test data with highly correlated features
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": [2, 4, 6, 8],  # feature2 = 2 * feature1
            "feature3": [10, 20, 30, 40],
        }
    )
    y = np.array([0, 1, 0, 1])

    correlated_features = [("feature1", "feature2", 1.0)]

    # Mock sklearn's PCA to ensure it's called correctly
    with patch("sklearn.decomposition.PCA") as mock_pca:
        # Set up mock for fit_transform to return a transformed array
        mock_pca_instance = MagicMock()
        mock_pca.return_value = mock_pca_instance
        mock_pca_instance.fit_transform.return_value = np.array(
            [[0.1], [0.2], [0.3], [0.4]]
        )

        # Preprocess with PCA option
        X_processed, y_processed = preprocessor.preprocess(
            data,
            y,
            handle_correlations="pca",
            correlated_features=correlated_features,
            all_features=data.columns.tolist(),
        )

        # Check PCA was called with correct arguments
        mock_pca.assert_called_once_with(n_components=1)
        mock_pca_instance.fit_transform.assert_called_once()

        # Check original features were dropped and new PCA feature was added
        assert "feature1" not in X_processed.columns
        assert "feature2" not in X_processed.columns
        assert (
            "pca_feature1_feature2" in X_processed.columns
            or "pca_feature2_feature1" in X_processed.columns
        )


def test_handle_correlations_none(preprocessor):
    """Test not handling correlations when option is 'none'."""
    # Create test data with correlated features
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": [2, 4, 6, 8],  # feature2 = 2 * feature1
            "feature3": [5, 6, 7, 8],
        }
    )
    y = np.array([0, 1, 0, 1])

    correlated_features = [("feature1", "feature2", 1.0)]

    # Preprocess without handling correlations
    X_processed, y_processed = preprocessor.preprocess(
        data,
        y,
        handle_correlations="none",
        correlated_features=correlated_features,
        all_features=data.columns.tolist(),
    )

    # Check that all features were retained
    assert "feature1" in X_processed.columns
    assert "feature2" in X_processed.columns
    assert "feature3" in X_processed.columns
    assert list(X_processed.columns) == ["feature1", "feature2", "feature3"]


def test_handle_multiple_correlation_pairs(preprocessor):
    """Test handling multiple correlation pairs correctly."""
    # Create test data with multiple correlated pairs
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": [2, 4, 6, 8],  # correlated with feature1
            "feature3": [5, 6, 7, 8],
            "feature4": [10, 12, 14, 16],  # correlated with feature3
        }
    )
    y = np.array([0, 1, 0, 1])

    correlated_features = [
        ("feature1", "feature2", 1.0),
        ("feature3", "feature4", 0.98),
    ]

    # Preprocess with drop_first option
    X_processed, y_processed = preprocessor.preprocess(
        data,
        y,
        handle_correlations="drop_first",
        correlated_features=correlated_features,
        all_features=data.columns.tolist(),
    )

    # Check that first feature in each pair was dropped
    assert "feature1" not in X_processed.columns
    assert "feature3" not in X_processed.columns
    assert "feature2" in X_processed.columns
    assert "feature4" in X_processed.columns
    assert list(X_processed.columns) == ["feature2", "feature4"]


def test_handle_overlapping_correlation_pairs(preprocessor):
    """Test handling overlapping correlation pairs correctly."""
    # Create test data with overlapping correlated pairs
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": [2, 4, 6, 8],  # correlated with feature1
            "feature3": [3, 6, 9, 12],  # correlated with both feature1 and feature2
        }
    )
    y = np.array([0, 1, 0, 1])

    correlated_features = [("feature1", "feature2", 1.0), ("feature2", "feature3", 1.0)]

    # Preprocess with drop_first option
    X_processed, y_processed = preprocessor.preprocess(
        data,
        y,
        handle_correlations="drop_first",
        correlated_features=correlated_features,
        all_features=data.columns.tolist(),
    )

    # Check that feature1 and feature2 were dropped (feature2 could be kept in the second pair
    # since feature1 was already dropped, but then feature2 gets dropped in its new first position)
    assert "feature1" not in X_processed.columns
    assert "feature2" not in X_processed.columns
    assert "feature3" in X_processed.columns
    assert list(X_processed.columns) == ["feature3"]


def test_integration_constant_and_correlation_handling(preprocessor):
    """Test integrated handling of both constant and correlated features."""
    # Create test data with both constant and correlated features
    data = pd.DataFrame(
        {
            "constant": [5, 5, 5, 5],
            "feature1": [1, 2, 3, 4],
            "feature2": [2, 4, 6, 8],  # correlated with feature1
            "feature3": [10, 20, 30, 40],
        }
    )
    y = np.array([0, 1, 0, 1])

    constant_features = ["constant"]
    correlated_features = [("feature1", "feature2", 1.0)]

    # Apply both constant and correlation handling
    X_processed, y_processed = preprocessor.preprocess(
        data,
        y,
        handle_constant_features="drop",
        handle_correlations="drop_lowest",
        constant_features=constant_features,
        correlated_features=correlated_features,
        all_features=data.columns.tolist(),
    )

    # Check that constant feature and the lower variance feature were dropped
    assert "constant" not in X_processed.columns
    assert "feature1" not in X_processed.columns  # Lower variance
    assert "feature2" in X_processed.columns
    assert "feature3" in X_processed.columns
    assert list(X_processed.columns) == ["feature2", "feature3"]


@patch("builtins.print")  # Mock print to avoid console output during tests
def test_preprocess_onehot_encoding(mock_print, preprocessor, sample_data):
    """Test one-hot encoding for categorical features"""
    X, y = sample_data
    X_processed, _ = preprocessor.preprocess(
        X,
        y,
        categorical_features={"cat1": "onehot"},
        all_features=["num1", "num2", "num3", "cat1"],
    )

    # Original column should be dropped
    assert "cat1" not in X_processed.columns

    # One-hot encoded columns should be present
    assert (
        "cat1_cat1" in X_processed.columns
        or "cat1_cat2" in X_processed.columns
        or "cat1_cat3" in X_processed.columns
    )

    # Check that binary columns were created
    onehot_cols = [col for col in X_processed.columns if col.startswith("cat1_")]
    assert len(onehot_cols) == 3  # 3 unique categories


@patch("builtins.print")  # Mock print to avoid console output during tests
def test_preprocess_hash_encoding(mock_print, preprocessor, categorical_data):
    """Test hash encoding for high cardinality features"""
    X = categorical_data
    y = np.array([0, 1, 0, 1, 0])

    X_processed, _ = preprocessor.preprocess(
        X,
        y,
        categorical_features={
            "low_cardinality": "onehot",
            "skewed_distribution": "label",
            "ordinal_feature": "ordinal",
            "high_cardinality": ["hash", 10],
        },  # 10 hash components
        all_features=X.columns.tolist(),
    )

    # Original column should be dropped
    assert "high_cardinality" not in X_processed.columns

    # Hash encoded columns should be present
    hash_cols = [
        col for col in X_processed.columns if col.startswith("high_cardinality_hash_")
    ]
    assert len(hash_cols) == 10  # 10 hash components


def test_preprocess_label_encoding(preprocessor, sample_data):
    """Test label encoding for categorical features"""
    X, y = sample_data
    X_processed, _ = preprocessor.preprocess(
        X,
        y,
        categorical_features={"cat1": "label"},
        all_features=["num1", "num2", "num3", "cat1"],
    )

    # Column should still exist but values should be encoded
    assert "cat1" in X_processed.columns
    assert X_processed["cat1"].dtype == np.int64
    assert set(X_processed["cat1"].unique()) == {0, 1, 2}  # 3 unique encoded values


def test_preprocess_ordinal_encoding(preprocessor, categorical_data):
    """Test ordinal encoding for categorical features"""
    X = categorical_data
    y = np.array([0, 1, 0, 1, 0])

    X_processed, _ = preprocessor.preprocess(
        X,
        y,
        categorical_features={
            "low_cardinality": "onehot",
            "skewed_distribution": "label",
            "ordinal_feature": "ordinal",
            "high_cardinality": ["hash", 10],
        },
        all_features=X.columns.tolist(),
    )

    # Column should still exist but values should be encoded
    assert "ordinal_feature" in X_processed.columns
    assert X_processed["ordinal_feature"].dtype == np.int64

    # Check ordering preservation (assuming "low" < "medium" < "high")
    # Get original ordinal values
    original_values = categorical_data["ordinal_feature"].tolist()
    encoded_values = X_processed["ordinal_feature"].tolist()

    # Check that the relative ordering is preserved
    for i in range(len(original_values)):
        for j in range(len(original_values)):
            if (
                (original_values[i] == "low" and original_values[j] == "medium")
                or (original_values[i] == "low" and original_values[j] == "high")
                or (original_values[i] == "medium" and original_values[j] == "high")
            ):
                assert encoded_values[i] < encoded_values[j]


def test_preprocess_multiple_encoding_types(preprocessor, categorical_data):
    """Test using multiple encoding types in the same preprocessing operation"""
    X = categorical_data
    y = np.array([0, 1, 0, 1, 0])

    encoding_dict = {
        "low_cardinality": "onehot",
        "high_cardinality": ["hash", 8],
        "skewed_distribution": "label",
        "ordinal_feature": "ordinal",
    }

    X_processed, _ = preprocessor.preprocess(
        X, y, categorical_features=encoding_dict, all_features=X.columns.tolist()
    )

    # Check one-hot encoding result
    assert "low_cardinality" not in X_processed.columns
    assert any(col.startswith("low_cardinality_") for col in X_processed.columns)

    # Check hash encoding result
    assert "high_cardinality" not in X_processed.columns
    hash_cols = [
        col for col in X_processed.columns if col.startswith("high_cardinality_hash_")
    ]
    assert len(hash_cols) == 8

    # Check label encoding result
    assert "skewed_distribution" in X_processed.columns
    assert X_processed["skewed_distribution"].dtype == np.int64

    # Check ordinal encoding result
    assert "ordinal_feature" in X_processed.columns
    assert X_processed["ordinal_feature"].dtype == np.int64


def test_preprocess_handle_missing_drop(preprocessor, sample_data):
    """Test handling missing values by dropping rows"""
    X, y = sample_data
    X_processed, y_processed = preprocessor.preprocess(
        X,
        y,
        handle_missing="drop",
        categorical_features={"cat1": "label"},
        all_features=["num1", "num2", "num3", "cat1"],
    )

    # Should have dropped one row with missing value
    assert len(X_processed) == 3
    assert len(y_processed) == 3
    assert not X_processed.isna().any().any()


def test_preprocess_target_encoding(preprocessor, sample_data):
    """Test encoding of the target variable"""
    X, y = sample_data
    _, y_processed = preprocessor.preprocess(
        X,
        y,
        encode_target=True,
        categorical_features={"cat1": "label"},
        all_features=["num1", "num2", "num3", "cat1"],
    )

    # Target should be encoded to integers
    assert y_processed.dtype == np.int64
    assert set(y_processed) == {0, 1}  # Binary encoded labels

    # Check that the label encoder was stored
    assert preprocessor.label_encoder is not None


def test_preprocess_without_target_encoding(preprocessor, sample_data):
    """Test skipping encoding of the target variable"""
    X, y = sample_data
    _, y_processed = preprocessor.preprocess(
        X,
        y,
        encode_target=False,
        categorical_features={"cat1": "label"},
        all_features=["num1", "num2", "num3", "cat1"],
    )

    # Target should remain as strings
    assert np.array_equal(y_processed, y)
    assert preprocessor.label_encoder is None


@patch("logging.warning")
def test_preprocess_high_cardinality_warning(mock_warning, preprocessor):
    """Test warning for high cardinality with one-hot encoding"""
    # Create high cardinality data
    high_card_data = pd.DataFrame(
        {"high_card": [f"val_{i}" for i in range(60)]}  # 60 unique values
    )
    y = np.zeros(60)

    with patch("builtins.print"):  # Suppress prints
        X_processed, _ = preprocessor.preprocess(
            high_card_data,
            y,
            categorical_features={"high_card": "onehot"},
            all_features=["high_card"],
        )

    # Should warn about high cardinality
    mock_warning.assert_called()

    # Check the exact call content
    warning_calls = [str(call) for call in mock_warning.call_args_list]
    assert any(
        "One-hot encoding may create too many features" in call
        for call in warning_calls
    )

    # Verify that one-hot encoding was applied
    # The original 'high_card' column should not be in X_processed anymore
    assert "high_card" not in X_processed.columns

    # Check that new one-hot encoded columns are present
    onehot_cols = [col for col in X_processed.columns if col.startswith("high_card_")]
    assert len(onehot_cols) == 60  # 60 unique values = 60 one-hot encoded columns

    # Check the column names to make sure they are correct
    for col in onehot_cols:
        assert col.startswith("high_card_")


@patch("logging.warning")
def test_preprocess_very_high_cardinality_fallback(mock_warning, preprocessor):
    """Test fallback to hash encoding for very high cardinality"""
    # Create very high cardinality data
    very_high_card_data = pd.DataFrame(
        {"very_high_card": [f"val_{i}" for i in range(120)]}  # 120 unique values
    )
    y = np.zeros(120)

    with patch("builtins.print"):  # Suppress prints
        X_processed, _ = preprocessor.preprocess(
            very_high_card_data,
            y,
            categorical_features={"very_high_card": "onehot"},
            all_features=["very_high_card"],
        )

    # Should warn about falling back to hash encoding
    mock_warning.assert_called()
    assert any(
        "Falling back to hash encoding" in str(call)
        for call in mock_warning.call_args_list
    )

    # Should have created hash columns
    hash_cols = [
        col for col in X_processed.columns if col.startswith("very_high_card_hash_")
    ]
    assert len(hash_cols) == 32  # Default hash component count


def test_assign_encoding_types_auto(preprocessor, categorical_data_for_auto_encoding):
    """Test automatic assignment of encoding types"""
    with patch("builtins.print"):  # Suppress prints
        encoding_types = preprocessor.assign_encoding_types(
            categorical_data_for_auto_encoding,
            [
                "low_cardinality",
                "high_cardinality",
                "skewed_distribution",
                "ordinal_feature",
                "many_rare_low_card",
            ],
            encoding_type="auto",
            ordinal_columns=["ordinal_feature"],
        )

    print(f"encoding_types: {encoding_types}")
    # Check that each column got a suitable encoding type
    assert encoding_types["low_cardinality"] == "onehot"  # Low cardinality
    assert isinstance(
        encoding_types["high_cardinality"], list
    )  # High cardinality, no heavy skew, and not heavy in rare values gets hash
    assert encoding_types["high_cardinality"][0] == "hash"
    assert (
        encoding_types["skewed_distribution"] == "label"
    )  # Skewed, with cardinality > 20 gets label
    assert encoding_types["ordinal_feature"] == "ordinal"  # Ordinal specified
    assert encoding_types["many_rare_low_card"] == "label"


def test_assign_encoding_types_all_onehot(preprocessor, categorical_data):
    """Test assigning the same encoding type to all columns"""
    encoding_types = preprocessor.assign_encoding_types(
        categorical_data,
        ["low_cardinality", "high_cardinality", "skewed_distribution"],
        encoding_type="onehot",
    )

    # All columns should have the same encoding type
    assert all(val == "onehot" for val in encoding_types.values())
    assert len(encoding_types) == 3


def test_assign_encoding_types_all_hash(preprocessor, categorical_data):
    """Test assigning hash encoding to all columns"""
    encoding_types = preprocessor.assign_encoding_types(
        categorical_data,
        ["low_cardinality", "high_cardinality"],
        encoding_type="hash",
        hash_components=16,
    )

    # All columns should have hash encoding with specified components
    assert all(val[0] == "hash" and val[1] == 16 for val in encoding_types.values())
    assert len(encoding_types) == 2


def test_assign_encoding_types_invalid_columns(preprocessor, categorical_data):
    """Test handling of invalid column names"""
    with patch("logging.warning") as mock_warning:
        encoding_types = preprocessor.assign_encoding_types(
            categorical_data,
            ["non_existent_column", "low_cardinality"],
            encoding_type="label",
        )

    # Should warn about invalid column
    mock_warning.assert_called_once()
    assert "not found" in mock_warning.call_args[0][0]

    # Should only include valid column
    assert len(encoding_types) == 1
    assert "low_cardinality" in encoding_types
    assert "non_existent_column" not in encoding_types


def test_assign_encoding_types_no_valid_columns(preprocessor, categorical_data):
    """Test handling when no valid columns are provided"""
    with patch("logging.warning") as mock_warning:
        encoding_types = preprocessor.assign_encoding_types(
            categorical_data,
            ["non_existent_column1", "non_existent_column2"],
            encoding_type="label",
        )

    # Should warn about no valid columns
    assert any(
        "No valid categorical columns" in str(call)
        for call in mock_warning.call_args_list
    )

    # Should return empty dictionary
    assert encoding_types == {}


def test_assign_encoding_types_invalid_type(preprocessor, categorical_data):
    """Test handling of invalid encoding type"""
    with pytest.raises(ValueError) as exc_info:
        preprocessor.assign_encoding_types(
            categorical_data, ["low_cardinality"], encoding_type="invalid_type"
        )

    # Should raise ValueError with message about invalid type
    assert "Invalid encoding type" in str(exc_info.value)
