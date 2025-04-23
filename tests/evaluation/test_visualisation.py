import pytest
import os
import sys
import plotly
import numpy as np
from unittest.mock import patch
from balancr.evaluation import (
    plot_class_distribution,
    plot_class_distributions_comparison,
    plot_comparison_results,
    plot_learning_curves,
    plot_radar_chart,
    plot_3d_scatter,
)


@pytest.fixture
def sample_distribution():
    """Create sample class distribution"""
    return {0: 800, 1: 200}


@pytest.fixture
def sample_distributions():
    """Create sample distributions for multiple techniques"""
    return {"SMOTE": {0: 500, 1: 500}, "RandomUnderSampler": {0: 200, 1: 200}}


@pytest.fixture
def sample_results():
    """Create sample comparison results with nested structure"""
    return {
        "RandomForestClassifier": {
            "SMOTE": {
                "standard_metrics": {
                    "precision": 0.8,
                    "recall": 0.7,
                    "f1": 0.75,
                    "roc_auc": 0.85,
                }
            },
            "RandomUnderSampler": {
                "standard_metrics": {
                    "precision": 0.75,
                    "recall": 0.8,
                    "f1": 0.775,
                    "roc_auc": 0.82,
                }
            },
        }
    }


@pytest.fixture
def sample_learning_curve_data():
    """Create sample learning curve data"""
    return {
        "SMOTE": {
            "train_sizes": np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
            "train_scores": np.random.rand(5, 3),
            "val_scores": np.random.rand(5, 3),
        }
    }


@pytest.fixture
def temp_path(tmp_path):
    """Create a temporary directory for saving plot files"""
    return tmp_path / "test_plots"


@patch("matplotlib.pyplot.show")
def test_plot_class_distribution(mock_show, sample_distribution, temp_path):
    """Test if plot_class_distribution runs without errors and saves files correctly"""
    plot_class_distribution(sample_distribution, display=True)
    mock_show.assert_called_once()
    mock_show.reset_mock()  # Reset the mock for the next call

    # Test with saving
    save_path = temp_path / "class_dist.png"
    os.makedirs(temp_path, exist_ok=True)
    plot_class_distribution(sample_distribution, save_path=str(save_path))
    assert save_path.exists()
    mock_show.assert_not_called()  # plt.show shouldn't be called when display=False


@patch("matplotlib.pyplot.show")
def test_plot_class_distributions_comparison(
    mock_show, sample_distributions, temp_path
):
    """Test if plot_class_distributions_comparison runs without errors and saves files correctly"""
    # Test with display=True
    plot_class_distributions_comparison(sample_distributions, display=True)
    mock_show.assert_called_once()
    mock_show.reset_mock()

    # Test with saving
    save_path = temp_path / "class_dist_comparison.png"
    os.makedirs(temp_path, exist_ok=True)
    plot_class_distributions_comparison(sample_distributions, save_path=str(save_path))
    assert save_path.exists()
    mock_show.assert_not_called()


@patch("matplotlib.pyplot.show")
def test_plot_comparison_results(mock_show, sample_results, temp_path):
    """Test if plot_comparison_results runs without errors and saves files correctly"""
    # Test with display=True
    plot_comparison_results(
        sample_results, classifier_name="RandomForestClassifier", display=True
    )
    mock_show.assert_called_once()
    mock_show.reset_mock()

    # Test with saving
    save_path = temp_path / "comparison_results.png"
    os.makedirs(temp_path, exist_ok=True)
    plot_comparison_results(
        sample_results,
        classifier_name="RandomForestClassifier",
        save_path=str(save_path),
    )
    assert save_path.exists()
    mock_show.assert_not_called()


@patch("matplotlib.pyplot.show")
def test_plot_learning_curves(mock_show, sample_learning_curve_data, temp_path):
    """Test if plot_learning_curves runs without errors and saves files correctly"""
    # Test with display=True
    plot_learning_curves(sample_learning_curve_data, display=True)
    mock_show.assert_called_once()
    mock_show.reset_mock()

    # Test with saving
    save_path = temp_path / "learning_curves.png"
    os.makedirs(temp_path, exist_ok=True)
    plot_learning_curves(sample_learning_curve_data, save_path=str(save_path))
    assert save_path.exists()
    mock_show.assert_not_called()


def test_plot_class_distribution_invalid_input():
    """Test if plot_class_distribution handles invalid input correctly"""
    with pytest.raises(TypeError):
        plot_class_distribution(None)

    with pytest.raises((TypeError, ValueError)):
        plot_class_distribution({})


def test_plot_class_distributions_comparison_invalid_input():
    """Test if plot_class_distributions_comparison handles invalid input correctly"""
    with pytest.raises(ValueError):
        plot_class_distributions_comparison({})

    with pytest.raises(TypeError):
        plot_class_distributions_comparison(None)


def test_plot_comparison_results_invalid_input():
    """Test if plot_comparison_results handles invalid input correctly"""
    with pytest.raises(ValueError):
        plot_comparison_results({}, classifier_name="NonExistentClassifier")

    with pytest.raises(TypeError):
        plot_comparison_results(None, classifier_name="SomeClassifier")


def test_plot_learning_curves_invalid_input():
    """Test if plot_learning_curves handles invalid input correctly"""
    with pytest.raises((ValueError, TypeError)):
        plot_learning_curves({})

    with pytest.raises(TypeError):
        plot_learning_curves(None)


@patch("matplotlib.pyplot.show")
def test_plot_radar_chart(mock_show, sample_results, temp_path):
    """Test if plot_radar_chart runs without errors and saves files correctly"""
    # Test with display=True
    plot_radar_chart(
        sample_results,
        classifier_name="RandomForestClassifier",
        metric_type="standard_metrics",
        display=True,
    )
    mock_show.assert_called_once()
    mock_show.reset_mock()

    # Test with saving
    save_path = temp_path / "radar_chart.png"
    os.makedirs(temp_path, exist_ok=True)
    plot_radar_chart(
        sample_results,
        classifier_name="RandomForestClassifier",
        metric_type="standard_metrics",
        save_path=str(save_path),
    )
    assert save_path.exists()
    mock_show.assert_not_called()

    # Test with custom metrics
    custom_metrics = ["precision", "recall", "f1"]
    save_path = temp_path / "custom_radar_chart.png"
    plot_radar_chart(
        sample_results,
        classifier_name="RandomForestClassifier",
        metric_type="standard_metrics",
        metrics_to_plot=custom_metrics,
        save_path=str(save_path),
    )
    assert save_path.exists()

    # Test with CV metrics
    cv_results = {
        "RandomForestClassifier": {
            "SMOTE": {
                "cv_metrics": {
                    "cv_accuracy_mean": 0.85,
                    "cv_accuracy_std": 0.05,
                    "cv_precision_mean": 0.83,
                    "cv_precision_std": 0.06,
                    "cv_recall_mean": 0.81,
                    "cv_recall_std": 0.07,
                    "cv_f1_mean": 0.82,
                    "cv_f1_std": 0.05,
                }
            },
            "RandomUnderSampler": {
                "cv_metrics": {
                    "cv_accuracy_mean": 0.80,
                    "cv_accuracy_std": 0.07,
                    "cv_precision_mean": 0.79,
                    "cv_precision_std": 0.08,
                    "cv_recall_mean": 0.82,
                    "cv_recall_std": 0.06,
                    "cv_f1_mean": 0.80,
                    "cv_f1_std": 0.07,
                }
            },
        }
    }

    save_path = temp_path / "cv_radar_chart.png"
    plot_radar_chart(
        cv_results,
        classifier_name="RandomForestClassifier",
        metric_type="cv_metrics",
        save_path=str(save_path),
    )
    assert save_path.exists()


def test_plot_radar_chart_invalid_input():
    """Test if plot_radar_chart handles invalid input correctly"""
    # Test with None results
    with pytest.raises(TypeError):
        plot_radar_chart(None, classifier_name="SomeClassifier")

    # Test with non-existent classifier
    with pytest.raises(ValueError):
        plot_radar_chart({}, classifier_name="NonExistentClassifier")

    # Test with no data for the specified metric type
    results = {"SomeClassifier": {"SomeTechnique": {"other_metrics": {}}}}
    with pytest.raises(ValueError):
        plot_radar_chart(results, classifier_name="SomeClassifier")

    # Test with no common metrics
    results = {
        "SomeClassifier": {
            "Technique1": {"standard_metrics": {"metric1": 0.8}},
            "Technique2": {"standard_metrics": {"metric2": 0.7}},
        }
    }
    with pytest.raises(ValueError):
        plot_radar_chart(
            results, classifier_name="SomeClassifier", metrics_to_plot=["metric3"]
        )


@pytest.fixture
def sample_3d_metrics_results():
    """Create sample results with metrics needed for 3D scatter plot"""
    return {
        "RandomForestClassifier": {
            "SMOTE": {
                "standard_metrics": {
                    "precision": 0.8,
                    "recall": 0.7,
                    "f1": 0.75,
                    "roc_auc": 0.85,
                    "g_mean": 0.76,
                },
                "cv_metrics": {
                    "cv_precision_mean": 0.78,
                    "cv_recall_mean": 0.72,
                    "cv_f1_mean": 0.74,
                    "cv_roc_auc_mean": 0.83,
                    "cv_g_mean_mean": 0.75,
                },
            },
            "RandomUnderSampler": {
                "standard_metrics": {
                    "precision": 0.75,
                    "recall": 0.8,
                    "f1": 0.775,
                    "roc_auc": 0.82,
                    "g_mean": 0.78,
                },
                "cv_metrics": {
                    "cv_precision_mean": 0.73,
                    "cv_recall_mean": 0.77,
                    "cv_f1_mean": 0.75,
                    "cv_roc_auc_mean": 0.81,
                    "cv_g_mean_mean": 0.76,
                },
            },
        },
        "LogisticRegression": {
            "SMOTE": {
                "standard_metrics": {
                    "precision": 0.72,
                    "recall": 0.65,
                    "f1": 0.68,
                    "roc_auc": 0.80,
                    "g_mean": 0.71,
                },
                "cv_metrics": {
                    "cv_precision_mean": 0.70,
                    "cv_recall_mean": 0.67,
                    "cv_f1_mean": 0.68,
                    "cv_roc_auc_mean": 0.79,
                    "cv_g_mean_mean": 0.72,
                },
            }
        },
    }


@patch("plotly.graph_objects.Figure.write_html")
@patch("plotly.graph_objects.Figure.show")
def test_plot_3d_scatter(
    mock_show, mock_write_html, sample_3d_metrics_results, temp_path
):
    """Test if plot_3d_scatter runs without errors and saves files correctly"""
    # Test with display=True for standard metrics
    plot_3d_scatter(
        sample_3d_metrics_results, metric_type="standard_metrics", display=True
    )
    mock_show.assert_called_once()
    mock_show.reset_mock()

    # Test with saving for standard metrics
    save_path = temp_path / "3d_scatter.html"
    plot_3d_scatter(
        sample_3d_metrics_results, metric_type="standard_metrics", save_path=save_path
    )
    mock_write_html.assert_called_once()
    mock_write_html.reset_mock()

    # Test with CV metrics
    plot_3d_scatter(
        sample_3d_metrics_results,
        metric_type="cv_metrics",
        save_path=temp_path / "3d_scatter_cv.html",
    )
    mock_write_html.assert_called_once()
    mock_write_html.reset_mock()
    mock_show.assert_not_called()


@patch("logging.warning")
def test_plot_3d_scatter_missing_metrics(mock_warning, sample_3d_metrics_results):
    """Test plot_3d_scatter handling when required metrics are missing"""
    # Create results with missing metrics
    incomplete_results = {
        "SomeClassifier": {
            "SomeTechnique": {
                "standard_metrics": {
                    "precision": 0.8,
                    "recall": 0.7,
                    # Missing f1, roc_auc, and g_mean
                }
            }
        }
    }

    # Call the function
    plot_3d_scatter(incomplete_results)

    # Verify both warnings were logged
    assert mock_warning.call_count == 2
    # First warning about the specific missing metrics
    assert "Cannot plot data point" in mock_warning.call_args_list[0][0][0]
    assert (
        "Missing metrics: ['f1', 'roc_auc', 'g_mean']"
        in mock_warning.call_args_list[0][0][0]
    )
    # Second warning about no valid data points
    assert (
        "No valid data points found for 3D scatter plot"
        in mock_warning.call_args_list[1][0][0]
    )


@patch("logging.warning")
def test_plot_3d_scatter_metrics_to_plot_filter(
    mock_warning, sample_3d_metrics_results
):
    """Test plot_3d_scatter when metrics_to_plot doesn't include required metrics"""
    # Call with metrics_to_plot that's missing required metrics
    plot_3d_scatter(
        sample_3d_metrics_results,
        metrics_to_plot=["precision", "recall"],  # Missing f1, roc_auc, g_mean
    )

    # Verify warning was logged
    mock_warning.assert_called_once()
    assert "Cannot create 3D scatter plot" in mock_warning.call_args[0][0]
    assert "Required metrics" in mock_warning.call_args[0][0]


@patch("plotly.graph_objects.Figure")
def test_plot_3d_scatter_empty_results(mock_figure, sample_3d_metrics_results):
    """Test plot_3d_scatter with empty results"""
    # Call with empty results
    plot_3d_scatter({})

    # Figure should not be created
    mock_figure.assert_not_called()


@pytest.mark.skipif("plotly" not in sys.modules, reason="Plotly is not installed")
def test_plot_3d_scatter_integration():
    """Integration test for 3D scatter plot (only runs if plotly is installed)"""
    plotly
    # Basic results to test actual plotly functionality
    test_results = {
        "TestClassifier": {
            "TestTechnique": {
                "standard_metrics": {"f1": 0.75, "roc_auc": 0.85, "g_mean": 0.76}
            }
        }
    }

    # This should run without error if plotly is installed
    fig = plot_3d_scatter(test_results)

    # Basic verification
    assert fig is not None
