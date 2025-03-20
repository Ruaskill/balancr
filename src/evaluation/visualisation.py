"""Visualization utilities for imbalanced data analysis."""

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

    # Prepare data for visualization
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
    results: Dict[str, Dict[str, float]],
    metrics_to_plot: Optional[list] = None,
    save_path: Optional[str] = None,
    display: bool = False,
) -> None:
    """
    Plot comparison of different techniques using various metrics.

    Args:
        results: Dictionary containing results for each technique
        metrics_to_plot: List of metrics to include in plot
        save_path: Path to save the plot
    """
    if results is None:
        raise TypeError("Results dictionary cannot be None")
    if not results or not isinstance(results, dict):
        raise ValueError("Results must be a non-empty dictionary")
    if not all(isinstance(d, dict) for d in results.values()):
        raise ValueError("Each result must be a dictionary of metrics")

    if metrics_to_plot is None:
        metrics_to_plot = ["precision", "recall", "f1", "roc_auc"]

    # Convert results to suitable format for plotting
    techniques = list(results.keys())
    metrics_data = {
        metric: [results[tech][metric] for tech in techniques]
        for metric in metrics_to_plot
    }

    # Create subplot for each metric
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(15, 10), squeeze=False)
    fig.suptitle("Comparison of Balancing Techniques", size=16)

    # Plot each metric
    for idx, (metric, values) in enumerate(metrics_data.items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        sns.barplot(x=techniques, y=values, ax=ax)
        ax.set_title(f'{metric.replace("_", " ").title()}')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Add value labels on top of bars
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")

    # Remove any empty subplots
    for idx in range(len(metrics_data), axes.shape[0] * axes.shape[1]):
        row = idx // 2
        col = idx % 2
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
