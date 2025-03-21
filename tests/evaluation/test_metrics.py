import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from evaluation.metrics import (
    get_metrics,
    get_cv_scores,
    get_learning_curve_data,
    get_learning_curve_data_multiple_techniques,
)


@pytest.fixture
def sample_data():
    """Create sample balanced dataset for testing"""
    np.random.seed(42)
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def imbalanced_data():
    """Create sample imbalanced dataset for testing"""
    np.random.seed(42)
    X = np.random.rand(100, 4)
    # Create imbalanced labels (80% class 0, 20% class 1)
    y = np.concatenate([np.zeros(80), np.ones(20)])
    return X, y


@pytest.fixture
def classifier():
    """Create a classifier instance for testing"""
    return RandomForestClassifier(random_state=42)


def test_get_metrics(sample_data, classifier):
    """Test if get_metrics returns correct metrics structure"""
    X, y = sample_data
    # Split data into train and test
    train_idx = np.random.choice(len(X), size=int(0.8 * len(X)), replace=False)
    test_idx = np.array([i for i in range(len(X)) if i not in train_idx])

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Train the classifier
    classifier.fit(X_train, y_train)

    metrics = get_metrics(classifier, X_train, y_train, X_test, y_test)

    # Check if all expected metrics are present
    expected_metrics = {
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "g_mean",
        "roc_auc",
        "average_precision",
    }
    assert set(metrics.keys()) == expected_metrics

    # Check if all metrics are float values between 0 and 1
    for metric_name, metric_value in metrics.items():
        if not np.isnan(metric_value):  # Skip NaN values that might appear
            assert isinstance(metric_value, float)
            assert 0 <= metric_value <= 1


def test_get_cv_scores(sample_data, classifier):
    """Test if get_cv_scores returns correct cross-validation scores structure"""
    X, y = sample_data
    cv_scores = get_cv_scores(classifier, X, y, n_folds=3)

    # Check if all expected metrics are present
    expected_metrics = {
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "cv_precision_mean",
        "cv_precision_std",
        "cv_recall_mean",
        "cv_recall_std",
        "cv_f1_mean",
        "cv_f1_std",
    }
    assert set(cv_scores.keys()) == expected_metrics

    # Check if all metrics are float values between 0 and 1
    for metric_value in cv_scores.values():
        assert isinstance(metric_value, float)
        assert 0 <= metric_value <= 1


def test_get_learning_curve_data(sample_data, classifier):
    """Test if get_learning_curve_data returns correct structure"""
    X, y = sample_data
    learning_curve_data = get_learning_curve_data(
        classifier, X, y, train_sizes=np.linspace(0.2, 1.0, 5)
    )

    # Check if all expected keys are present
    expected_keys = {"train_sizes", "train_scores", "val_scores"}
    assert set(learning_curve_data.keys()) == expected_keys

    # Check shapes of returned arrays
    n_splits = 5  # Default CV folds
    n_sizes = 5  # Number of train sizes that's specified
    assert learning_curve_data["train_sizes"].shape == (n_sizes,)
    assert learning_curve_data["train_scores"].shape == (n_sizes, n_splits)
    assert learning_curve_data["val_scores"].shape == (n_sizes, n_splits)


def test_get_learning_curve_data_multiple_techniques(sample_data, classifier):
    """Test if get_learning_curve_data_multiple_techniques returns correct structure"""
    X, y = sample_data

    # Create mock techniques data
    techniques_data = {
        "technique1": {"X_balanced": X, "y_balanced": y},
        "technique2": {"X_balanced": X, "y_balanced": y},
    }

    learning_curves = get_learning_curve_data_multiple_techniques(
        classifier, techniques_data, train_sizes=np.linspace(0.2, 1.0, 5)
    )

    # Check if all techniques are present
    assert set(learning_curves.keys()) == {"technique1", "technique2"}

    # Check structure for each technique
    for technique_data in learning_curves.values():
        assert set(technique_data.keys()) == {
            "train_sizes",
            "train_scores",
            "val_scores",
        }

        # Check shapes
        n_splits = 5  # Default CV folds
        n_sizes = 5  # Number of train sizes that's specified
        assert technique_data["train_sizes"].shape == (n_sizes,)
        assert technique_data["train_scores"].shape == (n_sizes, n_splits)
        assert technique_data["val_scores"].shape == (n_sizes, n_splits)
