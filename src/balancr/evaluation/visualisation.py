"""Visualisation utilities for imbalanced data analysis."""

from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pandas as pd


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

    fig.suptitle(
        title,
        size=16,
    )

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

        ax.set_title(f"{technique_name}")
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


def plot_radar_chart(
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    classifier_name: str,
    metric_type: str = "standard_metrics",
    metrics_to_plot: Optional[list] = None,
    save_path: Optional[str] = None,
    display: bool = False,
) -> None:
    """
    Plot radar (spider) chart comparing balancing techniques for a specific classifier.

    Args:
        results: Dictionary containing nested results structure
                 {classifier_name: {technique_name: {metric_type: {metric_name: value}}}}
        classifier_name: Name of the classifier to visualise
        metric_type: Type of metrics to plot ('standard_metrics' or 'cv_metrics')
        metrics_to_plot: List of metrics to include in the radar chart
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
            # For CV metrics, we're interested in mean values, not std
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
        # convert them to CV metric names with _mean suffix
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

    # Create the figure and polar axis
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Calculate angles for each metric (evenly spaced around the circle)
    angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()

    # Make the plot circular by appending the start point at the end
    angles += angles[:1]

    # Add metric labels
    metric_labels = []
    for metric in available_metrics:
        # Format the metric name for display
        if (
            metric_type == "cv_metrics"
            and metric.startswith("cv_")
            and metric.endswith("_mean")
        ):
            # For CV metrics, show "Metric Mean" format
            base_metric = metric[3:-5]  # Remove 'cv_' prefix and '_mean' suffix
            display_label = f'{base_metric.replace("_", " ").title()}'
        else:
            display_label = metric.replace("_", " ").title()
        metric_labels.append(display_label)

    # Set radar chart labels at appropriate angles
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)

    # Get color map for different techniques
    # Use matplotlib.colormaps instead of deprecated plt.cm.get_cmap
    import matplotlib as mpl

    if hasattr(mpl, "colormaps"):  # Matplotlib 3.7+
        cmap = mpl.colormaps["tab10"]
    else:  # Fallback for older versions
        cmap = plt.get_cmap("tab10")

    # Plot each technique
    for i, (technique_name, metrics) in enumerate(plot_data.items()):
        # Extract values for the metrics we want to plot
        values = [metrics[metric] for metric in available_metrics]

        # Make values circular
        values += values[:1]

        # Plot the technique
        color = cmap(i)
        ax.plot(angles, values, "o-", linewidth=2, label=technique_name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    # Set title
    title_type = (
        "Cross-Validation Metrics"
        if metric_type == "cv_metrics"
        else "Standard Metrics"
    )
    title = f"{classifier_name} - {title_type}"
    ax.set_title(title, size=15)

    # Add a legend
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    # Adjust layout
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if display:
        plt.show()

    plt.close()


def plot_3d_scatter(
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    metric_type: str = "standard_metrics",
    metrics_to_plot: Optional[list] = None,
    save_path: Optional[str] = None,
    display: bool = False,
) -> None:
    """
    Create a 3D scatter plot showing the relationship between F1-score, ROC-AUC, and G-mean
    for various classifier and balancing technique combinations.

    Args:
        results: Dictionary containing nested results structure
        metric_type: Type of metrics to plot ('standard_metrics' or 'cv_metrics')
        metrics_to_plot: List of metrics that were chosen to be plotted
        save_path: Path to save the plot
        display: Whether to display the plot
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        logging.error(
            "Plotly is required for 3D scatter plots. Install with: pip install plotly"
        )
        return

    # Determine metric keys based on the metric_type
    if metric_type == "standard_metrics":
        f1_key = "f1"
        roc_auc_key = "roc_auc"
        g_mean_key = "g_mean"
        title_prefix = "Standard Metrics"
    else:  # cv_metrics
        f1_key = "cv_f1_mean"
        roc_auc_key = "cv_roc_auc_mean"
        g_mean_key = "cv_g_mean_mean"
        title_prefix = "Cross-Validation Metrics"

    # Check if required metrics are in metrics_to_plot
    required_metrics = ["f1", "roc_auc", "g_mean"]
    if metrics_to_plot is not None:
        missing_metrics = [m for m in required_metrics if m not in metrics_to_plot]
        if missing_metrics:
            logging.warning(
                f"Cannot create 3D scatter plot. Required metrics {missing_metrics} are not in metrics_to_plot. "
                "Please include these metrics using 'configure-metrics'."
            )
            return

    # Prepare data for plotting
    plot_data = []

    # Assign a unique color to each classifier
    classifiers = list(results.keys())

    # Skip if no classifiers
    if not classifiers:
        logging.warning("No classifiers found in results.")
        return

    # Build the data structure for plotting
    for classifier_name in classifiers:
        classifier_results = results[classifier_name]

        for technique_name, technique_metrics in classifier_results.items():
            if metric_type in technique_metrics:
                metrics = technique_metrics[metric_type]

                # Check if all required metrics are available
                if all(key in metrics for key in [f1_key, roc_auc_key, g_mean_key]):
                    f1 = metrics[f1_key]
                    roc_auc = metrics[roc_auc_key]
                    g_mean = metrics[g_mean_key]

                    # Only add valid data points
                    if (
                        isinstance(f1, (int, float))
                        and isinstance(roc_auc, (int, float))
                        and isinstance(g_mean, (int, float))
                    ):

                        plot_data.append(
                            {
                                "classifier": classifier_name,
                                "technique": technique_name,
                                "f1": f1,
                                "roc_auc": roc_auc,
                                "g_mean": g_mean,
                            }
                        )
                else:
                    missing = [
                        k for k in [f1_key, roc_auc_key, g_mean_key] if k not in metrics
                    ]
                    logging.warning(
                        f"Cannot plot data point for {classifier_name}/{technique_name}. "
                        f"Missing metrics: {missing}"
                    )

    # Skip if no valid data points
    if not plot_data:
        logging.warning("No valid data points found for 3D scatter plot.")
        return

    # Convert to pandas DataFrame for easier processing
    df = pd.DataFrame(plot_data)

    # Create interactive 3D scatter plot
    fig = go.Figure()

    # Add traces for each classifier
    for classifier_name in df["classifier"].unique():
        subset = df[df["classifier"] == classifier_name]

        fig.add_trace(
            go.Scatter3d(
                x=subset["roc_auc"],  # X-axis: ROC-AUC
                y=subset["g_mean"],  # Y-axis: G-mean
                z=subset["f1"],  # Z-axis: F1-score
                mode="markers",
                marker=dict(
                    size=10,
                    opacity=0.8,
                ),
                name=classifier_name,
                text=[
                    f"Classifier: {row['classifier']}<br>Technique: {row['technique']}<br>"
                    f"F1: {row['f1']:.4f}<br>ROC-AUC: {row['roc_auc']:.4f}<br>G-mean: {row['g_mean']:.4f}"
                    for _, row in subset.iterrows()
                ],
                hoverinfo="text",
            )
        )

    # Update layout
    fig.update_layout(
        title=f"{title_prefix}: F1-score vs ROC-AUC vs G-mean",
        scene=dict(
            xaxis_title="ROC-AUC",
            yaxis_title="G-mean",
            zaxis_title="F1-score",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            zaxis=dict(range=[0, 1]),
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(
            x=0.01,
            y=0.99,
            traceorder="normal",
            font=dict(size=12),
        ),
        autosize=True,
        height=700,
    )

    # Save or display the figure
    if save_path:
        # Convert Path object to string if needed
        save_path_str = str(save_path)

        # Ensure .html extension for interactive plot
        if not save_path_str.endswith(".html"):
            save_path_str += ".html"

        # Save as interactive HTML
        fig.write_html(save_path_str)
        logging.info(f"3D scatter plot saved to {save_path_str}")

    if display:
        fig.show()

    return fig  # Return the figure object for potential further customisation
