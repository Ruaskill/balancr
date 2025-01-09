"""Metrics for evaluating imbalanced classification performance."""

from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier


def get_metrics(
    X_train_balanced: np.ndarray,
    y_train_balanced: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Calculate metrics specifically suited for imbalanced classification.

    Args:
        X_train_balanced: Balanced training features
        y_train_balanced: Balanced training labels
        X_test: Test features
        y_test: Test labels
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing various metric scores
    """
    # Train a classifier on balanced data
    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train_balanced, y_train_balanced)

    # Get predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Calculate confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Calculate specificity (true negative rate)
    specificity = tn / (tn + fp)

    # Calculate geometric mean
    g_mean = np.sqrt(recall_score(y_test, y_pred) * specificity)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "specificity": specificity,
        "f1": f1_score(y_test, y_pred),
        "g_mean": g_mean,
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "average_precision": average_precision_score(y_test, y_pred_proba),
    }


def get_cv_scores(
    X_balanced: np.ndarray,
    y_balanced: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Perform cross-validation and return average scores.

    Args:
        X_balanced: Balanced feature matrix
        y_balanced: Balanced target vector
        n_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing average metric scores
    """
    clf = RandomForestClassifier(random_state=random_state)

    # Calculate different metrics using cross-validation
    metrics = {}
    for metric in ["accuracy", "precision", "recall", "f1"]:
        scores = cross_val_score(
            clf, X_balanced, y_balanced, cv=n_folds, scoring=metric
        )
        metrics[f"cv_{metric}_mean"] = scores.mean()
        metrics[f"cv_{metric}_std"] = scores.std()

    return metrics


def get_learning_curve_data(
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
    n_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Compute data for plotting learning curves.

    Args:
        X: Feature matrix
        y: Target vector
        train_sizes: Relative or absolute sizes of the training dataset
        n_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing training sizes, training scores, and validation scores
    """
    clf = RandomForestClassifier(random_state=random_state)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator=clf,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=n_folds,
        scoring="accuracy",  # Default metric is accuracy
        random_state=random_state,
        shuffle=True,
    )

    return {
        "train_sizes": train_sizes_abs,
        "train_scores": train_scores,
        "val_scores": val_scores,
    }


def get_learning_curve_data_multiple_techniques(
    techniques_data: Dict[
        str, Dict[str, np.ndarray]
    ],
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
    n_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute data for plotting learning curves for multiple techniques.

    Args:
        techniques_data: A dictionary where keys are technique names and values are dictionaries
                          containing 'X_balanced' and 'y_balanced' for each technique
        train_sizes: Relative or absolute sizes of the training dataset
        n_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing training sizes, training scores, and validation scores for each technique
    """
    learning_curve_data = {}

    # Loop through each technique's data
    for technique_name, data in techniques_data.items():
        X_balanced = data["X_balanced"]
        y_balanced = data["y_balanced"]

        clf = RandomForestClassifier(random_state=random_state)

        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator=clf,
            X=X_balanced,
            y=y_balanced,
            train_sizes=train_sizes,
            cv=n_folds,
            scoring="accuracy",  # Default metric is accuracy
            random_state=random_state,
            shuffle=True,
        )

        learning_curve_data[technique_name] = {
            "train_sizes": train_sizes_abs,
            "train_scores": train_scores,
            "val_scores": val_scores,
        }

    return learning_curve_data
