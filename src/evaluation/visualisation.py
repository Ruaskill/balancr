"""Visualisation utilities for imbalanced data analysis."""

from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_class_distribution(
    distribution: Dict[Any, int],
    title: str = "Class Distribution",
    save_path: Optional[str] = None,
    display: bool = False,
) -> None:
    """Plot the distribution of classes in the dataset."""
    if distribution is None:
        raise TypeError("Distribution cannot be None")
    if not distribution or not isinstance(distribution, dict):
        raise ValueError("Distribution must be a non-empty dictionary")

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(distribution.keys()), y=list(distribution.values()))
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")

    # Add percentage labels on top of bars
    total = sum(distribution.values())
    for i, count in enumerate(distribution.values()):
        percentage = (count / total) * 100
        plt.text(i, count, f"{percentage:.1f}%", ha="center", va="bottom")

    if save_path:
        plt.savefig(save_path)

    if display:
        plt.show()

    plt.close()


def plot_class_distributions_comparison(
    distributions: Dict[str, Dict[Any, int]],
    title: str = "Class Distribution Comparison",
    save_path: Optional[str] = None,
    display: bool = False,
) -> None:
    """
    Compare class distributions across multiple techniques using bar plots.

    Args:
        distributions: Dictionary where keys are technique names and values are class distributions.
        title: Title for the plot.
        save_path: Path to save the plot (optional).

    Example input:
    {
        "SMOTE": {0: 500, 1: 500},
        "RandomUnderSampler": {0: 400, 1: 400},
    }
    """
    if distributions is None:
        raise TypeError("Distributions dictionary cannot be None")
    if not distributions or not isinstance(distributions, dict):
        raise ValueError("Distributions must be a non-empty dictionary")
    if not all(isinstance(d, dict) for d in distributions.values()):
        raise ValueError("Each distribution must be a dictionary")

    # Prepare data for visualisation
    techniques = []
    classes = []
    counts = []

    # Process each technique
    for technique, dist in distributions.items():
        for cls, count in dist.items():
            techniques.append(technique)
            classes.append(str(cls))  # Convert class label to string for plotting
            counts.append(count)

    # Create DataFrame for seaborn plotting
    import pandas as pd

    plot_data = pd.DataFrame(
        {"Technique": techniques, "Class": classes, "Count": counts}
    )

    # Plot grouped bar chart
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x="Class", y="Count", hue="Technique", data=plot_data)

    # Values for each bar
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
        )

    # Title and labels
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.legend(title="Technique")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)

    if display:
        plt.show()

    plt.close()


def plot_comparison_results(
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    classifier_name: str,
    metric_type: str = "standard_metrics",
    metrics_to_plot: Optional[list] = None,
    save_path: Optional[str] = None,
    display: bool = False,
) -> None:
    """
    Plot comparison of different techniques for a specific classifier and metric type.

    Args:
        results: Dictionary containing nested results structure
        classifier_name: Name of the classifier to visualise
        metric_type: Type of metrics to plot ('standard_metrics' or 'cv_metrics')
        metrics_to_plot: List of metrics to include in plot
        save_path: Path to save the plot
        display: Whether to display the plot
    """
    if results is None:
        raise TypeError("Results dictionary cannot be None")

    if classifier_name not in results:
        raise ValueError(f"Classifier '{classifier_name}' not found in results")

    # Extract the classifier's results
    classifier_results = results[classifier_name]

    # Create a structure for plotting with techniques as keys and metric dictionaries as values
    plot_data = {}
    for technique_name, technique_data in classifier_results.items():
        if metric_type in technique_data:
            plot_data[technique_name] = technique_data[metric_type]

    if not plot_data:
        raise ValueError(
            f"No {metric_type} data found for classifier '{classifier_name}'"
        )

    # Default metrics to plot
    if metrics_to_plot is None:
        if metric_type == "standard_metrics":
            metrics_to_plot = ["precision", "recall", "f1", "roc_auc"]
        elif metric_type == "cv_metrics":
            # For CV metrics, we want to look for metrics with "cv_" prefix and "_mean" suffix
            # But we want to use the same base metric names as configured
            metrics_to_plot = [
                "cv_accuracy_mean",
                "cv_precision_mean",
                "cv_recall_mean",
                "cv_f1_mean",
            ]
        else:
            metrics_to_plot = ["precision", "recall", "f1", "roc_auc"]
    elif (
        metric_type == "cv_metrics"
        and metrics_to_plot
        and not all(m.startswith("cv_") for m in metrics_to_plot)
    ):
        # If user provided standard metric names but we're plotting CV metrics,
        # convert them to CV metric names
        metrics_to_plot = [f"cv_{m}_mean" for m in metrics_to_plot]

    # Filter metrics to only include those that exist in all techniques
    common_metrics = set.intersection(
        *[set(metrics.keys()) for metrics in plot_data.values()]
    )
    available_metrics = [m for m in metrics_to_plot if m in common_metrics]

    if not available_metrics:
        # Show all available metrics in the error message
        raise ValueError(
            f"No common metrics found across techniques for metric type '{metric_type}'. "
            f"Requested metrics: {metrics_to_plot}, Available metrics: {sorted(common_metrics)}"
        )

    # Convert results to suitable format for plotting
    techniques = list(plot_data.keys())
    metrics_data = {
        metric: [plot_data[tech][metric] for tech in techniques]
        for metric in available_metrics
    }

    # Create subplot grid that accommodates all metrics
    n_metrics = len(available_metrics)
    n_cols = min(3, n_metrics)  # Maximum 3 columns to ensure readability
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False
    )
    fig.suptitle(
        f"{classifier_name} - Comparison of Balancing Techniques ({metric_type.replace('_', ' ').title()})",
        size=16,
    )

    # Plot each metric
    for idx, (metric, values) in enumerate(metrics_data.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        sns.barplot(x=techniques, y=values, ax=ax)

        # Set appropriate title based on metric type
        if metric.startswith("cv_") and metric.endswith("_mean"):
            # For CV metrics, show "Metric Mean" format
            base_metric = metric[3:-5]  # Remove 'cv_' prefix and '_mean' suffix
            display_title = f'{base_metric.replace("_", " ").title()} Mean'
        else:
            display_title = metric.replace("_", " ").title()

        ax.set_title(display_title)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Add value labels on top of bars
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")

    # Remove any empty subplots
    for idx in range(len(metrics_data), axes.shape[0] * axes.shape[1]):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if display:
        plt.show()

    plt.close()


def plot_learning_curves(
    learning_curve_data: dict,
    title: str = "Learning Curves",
    save_path: Optional[str] = None,
    display: bool = False,
):
    """
    Plot learning curves for multiple techniques in subplots.

    Args:
        learning_curve_data: Dictionary with technique names as keys and corresponding learning curve data as values
        title: Title of the plot
        save_path: Optional path to save the figure
    """
    # Error handling
    if learning_curve_data is None:
        raise TypeError("Learning curve data cannot be None")
    if not learning_curve_data or not isinstance(learning_curve_data, dict):
        raise ValueError("Learning curve data must be a non-empty dictionary")

    for technique, data in learning_curve_data.items():
        required_keys = {"train_sizes", "train_scores", "val_scores"}
        if not all(key in data for key in required_keys):
            raise ValueError(
                f"Learning curve data for technique '{technique}' must contain "
                f"'train_sizes', 'train_scores', and 'val_scores'"
            )

    num_techniques = len(learning_curve_data)
    # Set up a grid of subplots, one for each technique
    fig, axes = plt.subplots(num_techniques, 1, figsize=(10, 6 * num_techniques))

    # Ensure axes is iterable even if there's only one technique
    if num_techniques == 1:
        axes = [axes]

    for idx, (technique_name, data) in enumerate(learning_curve_data.items()):
        # Extract the train_sizes, train_scores, and val_scores from the dictionary
        train_sizes = data["train_sizes"]
        train_scores = data["train_scores"]
        val_scores = data["val_scores"]

        # Calculate mean and std of scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        ax = axes[idx]

        ax.plot(train_sizes, train_mean, label="Training score", color="blue")
        ax.plot(train_sizes, val_mean, label="Validation score", color="red")
        ax.fill_between(
            train_sizes,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.1,
            color="blue",
        )
        ax.fill_between(
            train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="red"
        )

        ax.set_title(f"{technique_name} - Learning Curves")
        ax.set_xlabel("Training Examples")
        ax.set_ylabel("Score")
        ax.legend(loc="best")
        ax.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if display:
        plt.show()

    plt.close()
