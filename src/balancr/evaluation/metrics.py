"""Metrics for evaluating imbalanced classification performance."""

import logging
import time
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, learning_curve


def format_time(seconds):
    """Format time in seconds to minutes and seconds"""
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}mins, {remaining_seconds:.2f}secs"


def get_metrics(
    classifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate metrics specifically suited for imbalanced classification.
    Works with both binary and multiclass problems.

    Args:
        classifier: Pre-fitted classifier instance to evaluate
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary containing various metric scores
    """
    # Get predictions
    y_pred = classifier.predict(X_test)

    metrics = {}

    # For ROC AUC, we need probability predictions
    if hasattr(classifier, "predict_proba"):
        try:
            y_pred_proba = classifier.predict_proba(X_test)
            # For multiclass problems, we'll use different methods below
        except (AttributeError, IndexError):
            # Log warning and set probability metrics to NaN
            logging.warning(
                f"Failed to get probability predictions from classifier {classifier.__class__.__name__}. "
                "ROC-AUC and average_precision metrics will be set to NaN."
            )
            metrics["roc_auc"] = float("nan")
            metrics["average_precision"] = float("nan")
            # Skip the rest of probability-based calculations
            y_pred_proba = None
    else:
        # Log warning and set probability metrics to NaN
        logging.warning(
            f"Classifier {classifier.__class__.__name__} does not support predict_proba. "
            "ROC-AUC and average_precision metrics will be set to NaN."
        )
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")
        # Skip the rest of probability-based calculations
        y_pred_proba = None

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    n_classes = len(np.unique(y_test))

    # Calculate metrics based on number of classes
    if n_classes == 2:  # Binary classification
        # Unpack binary confusion matrix
        tn, fp, fn, tp = cm.ravel()

        # Calculate binary metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        g_mean = np.sqrt(recall_score(y_test, y_pred) * specificity)

        # Create metrics dictionary for binary case
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
            # For binary classification, we need the probability of the positive class
            if (
                isinstance(y_pred_proba, np.ndarray)
                and y_pred_proba.ndim > 1
                and y_pred_proba.shape[1] > 1
            ):
                y_pred_proba_pos = y_pred_proba[:, 1]
            else:
                y_pred_proba_pos = y_pred_proba

            metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba_pos)
            metrics["average_precision"] = average_precision_score(
                y_test, y_pred_proba_pos
            )
        except (ValueError, TypeError):
            # Skip these metrics if they can't be calculated
            metrics["roc_auc"] = float("nan")
            metrics["average_precision"] = float("nan")

    else:  # Multiclass classification
        # Calculate per-class specificity and g-mean
        specificities = []
        recalls = []

        for i in range(n_classes):
            # For each class, treat it as positive and all others as negative
            y_true_binary = (y_test == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)

            # Calculate per-class CM values
            cm_i = confusion_matrix(y_true_binary, y_pred_binary)
            if cm_i.shape[0] < 2:  # Handle edge case
                specificities.append(0)
                recalls.append(0)
                continue

            tn_i, fp_i, fn_i, tp_i = cm_i.ravel()

            # Calculate specificity and recall for this class
            spec_i = tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0
            rec_i = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0

            specificities.append(spec_i)
            recalls.append(rec_i)

        # Calculate macro-averaged metrics
        macro_specificity = np.mean(specificities)
        macro_recall = np.mean(recalls)
        g_mean = np.sqrt(macro_recall * macro_specificity)

        # Create metrics dictionary for multiclass case
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="macro"),
            "recall": recall_score(y_test, y_pred, average="macro"),
            "specificity": macro_specificity,
            "f1": f1_score(y_test, y_pred, average="macro"),
            "g_mean": g_mean,
        }

        # Add multiclass ROC AUC if possible
        try:
            # For multiclass, use roc_auc_score with multi_class parameter
            if hasattr(classifier, "predict_proba"):
                metrics["roc_auc"] = roc_auc_score(
                    y_test, y_pred_proba, multi_class="ovr", average="macro"
                )

                # Calculate average precision for multiclass
                metrics["average_precision"] = precision_score(
                    y_test, y_pred, average="macro"
                )
            else:
                metrics["roc_auc"] = float("nan")
                metrics["average_precision"] = float("nan")
        except (ValueError, TypeError):
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
    Automatically handles multiclass data by using macro averaging.

    Args:
        classifier: Classifier instance to evaluate
        X_balanced: Balanced feature matrix
        y_balanced: Balanced target vector
        n_folds: Number of cross-validation folds

    Returns:
        Dictionary containing average metric scores
    """
    import logging
    from sklearn.model_selection import cross_val_score, cross_val_predict
    from sklearn.metrics import roc_auc_score, confusion_matrix
    import numpy as np

    # Determine if we're dealing with multiclass data
    unique_classes = np.unique(y_balanced)
    is_multiclass = len(unique_classes) > 2

    # Initialise metrics dictionary
    metrics = {}

    # Use scikit-learn's cross_val_score for standard metrics
    # Accuracy doesn't need special handling for multiclass
    scores = cross_val_score(
        classifier, X_balanced, y_balanced, cv=n_folds, scoring="accuracy"
    )
    metrics["cv_accuracy_mean"] = scores.mean()
    metrics["cv_accuracy_std"] = scores.std()

    # For metrics that need proper multiclass handling
    for metric in ["precision", "recall", "f1"]:
        # Use macro averaging for multiclass problems
        if is_multiclass:
            scoring = f"{metric}_macro"
        else:
            scoring = metric

        scores = cross_val_score(
            classifier, X_balanced, y_balanced, cv=n_folds, scoring=scoring
        )
        metrics[f"cv_{metric}_mean"] = scores.mean()
        metrics[f"cv_{metric}_std"] = scores.std()

    # For ROC-AUC and G-mean, we need the predictions
    try:
        # Get cross-validated predictions
        y_pred = cross_val_predict(classifier, X_balanced, y_balanced, cv=n_folds)

        # Calculate G-mean
        if is_multiclass:
            # Calculate per-class specificities and recalls for each class
            specificities = []
            recalls = []

            for i in unique_classes:
                # For each class, treat it as positive and all others as negative
                y_true_binary = (y_balanced == i).astype(int)
                y_pred_binary = (y_pred == i).astype(int)

                # Calculate confusion matrix
                cm = confusion_matrix(y_true_binary, y_pred_binary)

                # Ensure the confusion matrix has the right shape
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()

                    # Calculate specificity and recall
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                    specificities.append(specificity)
                    recalls.append(recall)

            # Calculate macro-averaged G-mean
            if specificities and recalls:  # Check if lists are not empty
                macro_specificity = np.mean(specificities)
                macro_recall = np.mean(recalls)
                g_mean = np.sqrt(macro_specificity * macro_recall)

                metrics["cv_g_mean_mean"] = g_mean
                metrics["cv_g_mean_std"] = (
                    0.0  # Cannot calculate std from a single value
                )
            else:
                logging.warning(
                    f"Could not calculate G-mean for classifier {classifier.__class__.__name__}. "
                    "Some classes may not have been predicted correctly."
                )
                # We don't set the metrics here, letting the absence indicate an issue
        else:
            # Binary case
            cm = confusion_matrix(y_balanced, y_pred)

            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()

                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

                g_mean = np.sqrt(specificity * sensitivity)
                metrics["cv_g_mean_mean"] = g_mean
            else:
                logging.warning(
                    f"Could not calculate G-mean for classifier {classifier.__class__.__name__}. "
                    "Unexpected confusion matrix shape."
                )
                # Metrics not set
    except Exception as e:
        logging.warning(
            f"Could not calculate G-mean for classifier {classifier.__class__.__name__}. "
            f"Error: {str(e)}"
        )
        # Metrics not set

    # ROC-AUC calculation - separate try block to ensure G-mean calculation happens even if ROC-AUC fails
    try:
        if hasattr(classifier, "predict_proba"):
            # Get probability predictions
            if is_multiclass:
                y_proba = cross_val_predict(
                    classifier,
                    X_balanced,
                    y_balanced,
                    cv=n_folds,
                    method="predict_proba",
                )
                roc_auc = roc_auc_score(
                    y_balanced, y_proba, multi_class="ovr", average="macro"
                )
            else:
                y_proba = cross_val_predict(
                    classifier,
                    X_balanced,
                    y_balanced,
                    cv=n_folds,
                    method="predict_proba",
                )
                # Use second column for positive class probability
                if y_proba.shape[1] > 1:
                    roc_auc = roc_auc_score(y_balanced, y_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_balanced, y_proba)

            metrics["cv_roc_auc_mean"] = roc_auc
        else:
            logging.warning(
                f"Classifier {classifier.__class__.__name__} does not support predict_proba. "
                "ROC-AUC cannot be calculated."
            )
            # Metrics not set
    except Exception as e:
        logging.warning(
            f"Could not calculate ROC-AUC for classifier {classifier.__class__.__name__}. "
            f"Error: {str(e)}"
        )
        # Metrics not set

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
    classifier_name: str,
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

        start_time = time.time()
        logging.info(
            f"Generating learning curve for {classifier_name} trained on data "
            f"balanced by {technique_name}, against balanced data..."
        )
        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator=classifier,
            X=X_balanced,
            y=y_balanced,
            train_sizes=train_sizes,
            cv=n_folds,
            scoring="accuracy",  # Default metric is accuracy
            shuffle=True,
        )
        curve_generating_time = time.time() - start_time
        logging.info(
            f"Generated learning curve for {classifier_name} trained on data "
            f"balanced by {technique_name}, against balanced data"
            f"(Time Taken: {format_time(curve_generating_time)})"
        )

        learning_curve_data[technique_name] = {
            "train_sizes": train_sizes_abs,
            "train_scores": train_scores,
            "val_scores": val_scores,
        }

    return learning_curve_data


def get_learning_curve_data_against_imbalanced_multiple_techniques(
    classifier_name: str,
    classifier,
    techniques_data: Dict[str, Dict[str, np.ndarray]],
    X_test,
    y_test,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
    n_folds: int = 5,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Custom learning curve function that trains on balanced data
    and evaluates on original imbalanced test set.

    Returns training and validation scores at each learning curve point.
    Validation scores are computed on X_test / y_test.
    """

    def safe_index(X, indices):
        return X.iloc[indices] if isinstance(X, (pd.DataFrame, pd.Series)) else X[indices]

    learning_curve_data = {}

    for technique_name, data in techniques_data.items():
        X_balanced = data["X_balanced"]
        y_balanced = data["y_balanced"]

        n_samples = X_balanced.shape[0]
        train_sizes_abs = (train_sizes * n_samples).astype(int)

        train_scores = []
        val_scores = []

        start_time = time.time()
        logging.info(
            f"Generating learning curve for {classifier_name} trained on data "
            f"balanced by {technique_name}, against original test data..."
        )

        for train_size in train_sizes_abs:
            X_subset = safe_index(X_balanced, np.arange(train_size))
            y_subset = safe_index(y_balanced, np.arange(train_size))

            fold_train_scores = []
            fold_val_scores = []

            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

            for train_idx, _ in kf.split(X_subset, y_subset):
                X_fold_train = safe_index(X_subset, train_idx)
                y_fold_train = safe_index(y_subset, train_idx)

                clf = clone(classifier)
                clf.fit(X_fold_train, y_fold_train)

                train_acc = clf.score(X_fold_train, y_fold_train)
                val_acc = clf.score(X_test, y_test)

                fold_train_scores.append(train_acc)
                fold_val_scores.append(val_acc)

            train_scores.append(fold_train_scores)
            val_scores.append(fold_val_scores)

        curve_generating_time = time.time() - start_time
        logging.info(
            f"Generated learning curve for {classifier_name} trained on data "
            f"balanced by {technique_name}, against original test data "
            f"(Time Taken: {format_time(curve_generating_time)})"
        )

        learning_curve_data[technique_name] = {
            "train_sizes": train_sizes_abs,
            "train_scores": np.array(train_scores),
            "val_scores": np.array(val_scores),
        }

    return learning_curve_data
