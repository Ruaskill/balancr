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


def get_metrics(
    classifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate metrics specifically suited for imbalanced classification.

    Args:
        classifier: Pre-fitted classifier instance to evaluate
        X_train_balanced: Balanced training features
        y_train_balanced: Balanced training labels
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary containing various metric scores
    """
    # Get predictions
    y_pred = classifier.predict(X_test)

    # For ROC AUC, we need probability predictions
    if hasattr(classifier, "predict_proba"):
        try:
            y_pred_proba = classifier.predict_proba(X_test)
            # Get probability for the positive class
            # Assuming binary classification for now
            if y_pred_proba.shape[1] > 1:
                y_pred_proba = y_pred_proba[:, 1]
            else:
                y_pred_proba = y_pred_proba.ravel()
        except (AttributeError, IndexError):
            # Fall back to binary predictions if probabilities fail
            y_pred_proba = y_pred
    else:
        # Use predicted classes if predict_proba isn't available
        y_pred_proba = y_pred

    # Calculate confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Calculate specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Calculate geometric mean
    g_mean = np.sqrt(recall_score(y_test, y_pred) * specificity)

    # Create metrics dictionary
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "specificity": specificity,
        "f1": f1_score(y_test, y_pred),
        "g_mean": g_mean,
    }

    # Add probability-based metrics if possible
    try:
        metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
        metrics["average_precision"] = average_precision_score(y_test, y_pred_proba)
    except ValueError:
        # Skip these metrics if they can't be calculated
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")

    return metrics


def get_cv_scores(
    classifier,
    X_balanced: np.ndarray,
    y_balanced: np.ndarray,
    n_folds: int = 5,
) -> Dict[str, float]:
    """
    Perform cross-validation and return average scores.

    Args:
        classifier: Classifier instance to evaluate
        X_balanced: Balanced feature matrix
        y_balanced: Balanced target vector
        n_folds: Number of cross-validation folds

    Returns:
        Dictionary containing average metric scores
    """
    # Calculate different metrics using cross-validation
    metrics = {}
    for metric in ["accuracy", "precision", "recall", "f1"]:
        scores = cross_val_score(
            classifier, X_balanced, y_balanced, cv=n_folds, scoring=metric
        )
        metrics[f"cv_{metric}_mean"] = scores.mean()
        metrics[f"cv_{metric}_std"] = scores.std()

    return metrics


def get_learning_curve_data(
    classifier,
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
    n_folds: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Compute data for plotting learning curves.

    Args:
        classifier: Classifier instance to evaluate
        X: Feature matrix
        y: Target vector
        train_sizes: Relative or absolute sizes of the training dataset
        n_folds: Number of cross-validation folds

    Returns:
        Dictionary containing training sizes, training scores, and validation scores
    """
    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator=classifier,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=n_folds,
        scoring="accuracy",  # Default metric is accuracy
        shuffle=True,
    )

    return {
        "train_sizes": train_sizes_abs,
        "train_scores": train_scores,
        "val_scores": val_scores,
    }


def get_learning_curve_data_multiple_techniques(
    classifier,
    techniques_data: Dict[str, Dict[str, np.ndarray]],
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
    n_folds: int = 5,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute data for plotting learning curves for multiple techniques.

    Args:
        classifier: Classifier instance to evaluate
        techniques_data: A dictionary where keys are technique names and values are dictionaries
                          containing 'X_balanced' and 'y_balanced' for each technique
        train_sizes: Relative or absolute sizes of the training dataset
        n_folds: Number of cross-validation folds

    Returns:
        Dictionary containing training sizes, training scores, and validation scores for each technique
    """
    learning_curve_data = {}

    # Loop through each technique's data
    for technique_name, data in techniques_data.items():
        X_balanced = data["X_balanced"]
        y_balanced = data["y_balanced"]

        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator=classifier,
            X=X_balanced,
            y=y_balanced,
            train_sizes=train_sizes,
            cv=n_folds,
            scoring="accuracy",  # Default metric is accuracy
            shuffle=True,
        )

        learning_curve_data[technique_name] = {
            "train_sizes": train_sizes_abs,
            "train_scores": train_scores,
            "val_scores": val_scores,
        }

    return learning_curve_data
