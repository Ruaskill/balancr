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

    metrics = get_metrics(classifier, X_test, y_test)

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


def test_get_metrics_binary_classification():
    """Test get_metrics for binary classification problems"""
    # Create sample binary data
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)

    # Split data into train and test
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    # Train a classifier
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # Get metrics
    metrics = get_metrics(classifier, X_test, y_test)

    # Check binary classification metrics
    binary_metrics = {
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "g_mean",
        "roc_auc",
        "average_precision",
    }

    assert set(metrics.keys()) == binary_metrics

    # Check that all values are floats between 0 and 1
    for metric_name, value in metrics.items():
        if not np.isnan(value):  # Some metrics might be NaN if not applicable
            assert isinstance(value, float)
            assert 0 <= value <= 1


def test_get_metrics_multiclass_classification():
    """Test get_metrics for multiclass classification problems"""
    # Create sample multiclass data
    X = np.random.rand(150, 4)
    y = np.random.randint(0, 3, 150)  # 3 classes

    # Split data
    X_train, X_test = X[:120], X[120:]
    y_train, y_test = y[:120], y[120:]

    # Train a classifier
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # Get metrics
    metrics = get_metrics(classifier, X_test, y_test)

    # Check for all the expected metrics
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

    # Check that all values are floats between 0 and 1
    for metric_name, value in metrics.items():
        if not np.isnan(value):  # Some metrics might be NaN if not applicable
            assert isinstance(value, float)
            assert 0 <= value <= 1


@pytest.mark.filterwarnings("ignore:Precision is ill-defined:sklearn.exceptions.UndefinedMetricWarning")
def test_get_metrics_with_classifier_without_proba():
    """Test get_metrics with a classifier that doesn't support predict_proba"""

    # Create a custom minimal classifier that doesn't have predict_proba
    class MinimalClassifier:
        def __init__(self):
            self.classes_ = [0, 1]

        def predict(self, X):
            # Always predict class 0
            return np.zeros(len(X))

        def fit(self, X, y):
            # Do nothing
            return self

    # Create sample data
    X = np.random.rand(50, 2)
    y = np.random.randint(0, 2, 50)

    # Create and train the classifier
    classifier = MinimalClassifier()
    classifier.fit(X, y)

    # Get metrics
    metrics = get_metrics(classifier, X[:10], y[:10])

    # Check that basic metrics are present
    basic_metrics = {"accuracy", "precision", "recall", "specificity", "f1", "g_mean"}
    assert all(metric in metrics for metric in basic_metrics)

    # Check that probability-based metrics are NaN
    prob_metrics = {"roc_auc", "average_precision"}
    for metric in prob_metrics:
        assert metric in metrics
        assert np.isnan(metrics[metric])


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
        "cv_roc_auc_mean",
        "cv_g_mean_mean",
    }
    assert set(cv_scores.keys()) == expected_metrics

    # Check if all metrics are float values between 0 and 1
    for metric_value in cv_scores.values():
        assert isinstance(metric_value, float)
        assert 0 <= metric_value <= 1


def test_get_cv_scores_enhanced_metrics(sample_data, classifier):
    """Test if get_cv_scores returns enhanced metrics like g-mean and ROC-AUC"""
    X, y = sample_data
    cv_scores = get_cv_scores(classifier, X, y, n_folds=3)

    # Check if enhanced metrics are present (or at least some of them)
    enhanced_metrics = {
        "cv_g_mean_mean",
        "cv_roc_auc_mean",
    }

    # At least one of the enhanced metrics should be present
    # (some might be missing if the classifier doesn't support predict_proba)
    assert any(metric in cv_scores for metric in enhanced_metrics)

    # Check specific values for metrics that should always be present
    for metric in [
        "cv_accuracy_mean",
        "cv_precision_mean",
        "cv_recall_mean",
        "cv_f1_mean",
    ]:
        assert metric in cv_scores
        assert isinstance(cv_scores[metric], float)
        assert 0 <= cv_scores[metric] <= 1


def test_get_cv_scores_binary_classification():
    """Test if get_cv_scores works correctly for binary classification"""
    # Create a simple binary classification dataset
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)  # Binary labels (0, 1)

    classifier = RandomForestClassifier(random_state=42)
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
        "cv_g_mean_mean",
        "cv_roc_auc_mean",
    }

    # Check that at least the basic metrics are present
    assert set(cv_scores.keys()) >= expected_metrics - {
        "cv_g_mean_mean",
        "cv_roc_auc_mean",
    }

    # G-mean and ROC-AUC should be present for a classifier that supports predict_proba
    assert "cv_g_mean_mean" in cv_scores
    assert "cv_roc_auc_mean" in cv_scores


def test_get_cv_scores_multiclass_classification():
    """Test if get_cv_scores works correctly for multiclass classification"""
    # Create a simple multiclass classification dataset
    X = np.random.rand(150, 4)
    y = np.random.randint(0, 3, 150)  # Multiclass labels (0, 1, 2)

    classifier = RandomForestClassifier(random_state=42)
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

    assert set(cv_scores.keys()) >= expected_metrics

    # Check for multiclass ROC-AUC, which should be present for RandomForest
    assert "cv_roc_auc_mean" in cv_scores

    # G-mean might or might not be present, depending on if all classes were predicted
    if "cv_g_mean_mean" in cv_scores:
        assert 0 <= cv_scores["cv_g_mean_mean"] <= 1


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
        "RandomForestClassifier",
        classifier,
        techniques_data,
        train_sizes=np.linspace(0.2, 1.0, 5),
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
