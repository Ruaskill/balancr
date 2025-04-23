import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from balancr.evaluation import get_metrics, get_cv_scores


@pytest.fixture
def binary_imbalanced_dataset():
    """Create a synthetic binary imbalanced dataset"""
    X, y = make_classification(
        n_samples=1000,
        n_classes=2,
        weights=[0.9, 0.1],  # Class imbalance ratio
        n_features=10,
        random_state=42,
    )
    return X, y


@pytest.fixture
def multiclass_imbalanced_dataset():
    """Create a synthetic multiclass imbalanced dataset"""
    X, y = make_classification(
        n_samples=1000,
        n_classes=3,
        n_informative=5,
        n_redundant=2,
        weights=[0.7, 0.2, 0.1],  # Class imbalance ratio
        n_features=10,
        random_state=42,
    )
    return X, y


def test_random_forest_with_smote_binary(binary_imbalanced_dataset):
    """Test metrics for RandomForest with SMOTE on binary data"""
    X, y = binary_imbalanced_dataset

    # Apply SMOTE balancing
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)

    # Split data
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42
    )

    # Train RandomForest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # Get metrics
    metrics = get_metrics(rf, X_test, y_test)

    # Check expected metrics range based on this known combination
    # These ranges are based on typical performance of RF+SMOTE on binary data
    assert 0.8 <= metrics["accuracy"] <= 1.0
    assert 0.8 <= metrics["precision"] <= 1.0
    assert 0.8 <= metrics["recall"] <= 1.0
    assert 0.8 <= metrics["f1"] <= 1.0
    assert 0.8 <= metrics["roc_auc"] <= 1.0
    assert 0.8 <= metrics["g_mean"] <= 1.0

    # Also check cross-validation scores
    cv_scores = get_cv_scores(
        RandomForestClassifier(random_state=42), X_balanced, y_balanced, n_folds=5
    )

    assert 0.8 <= cv_scores["cv_accuracy_mean"] <= 1.0
    assert 0.8 <= cv_scores["cv_precision_mean"] <= 1.0
    assert 0.8 <= cv_scores["cv_recall_mean"] <= 1.0
    assert 0.8 <= cv_scores["cv_f1_mean"] <= 1.0
