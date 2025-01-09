import pytest
import os
import numpy as np
from unittest.mock import patch
from evaluation.visualisation import (
    plot_class_distribution,
    plot_class_distributions_comparison,
    plot_comparison_results,
    plot_learning_curves,
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
    """Create sample comparison results"""
    return {
        "SMOTE": {
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.75,
            "roc_auc": 0.85
            },
        "RandomUnderSampler": {
            "precision": 0.75,
            "recall": 0.8,
            "f1": 0.775,
            "roc_auc": 0.82,
        },
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
    # Test without saving
    plot_class_distribution(sample_distribution)
    mock_show.assert_called_once()

    # Test with saving
    save_path = temp_path / "class_dist.png"
    os.makedirs(temp_path, exist_ok=True)
    plot_class_distribution(sample_distribution, save_path=str(save_path))
    assert save_path.exists()


@patch("matplotlib.pyplot.show")
def test_plot_class_distributions_comparison(
    mock_show, sample_distributions, temp_path
):
    """Test if plot_class_distributions_comparison runs without errors and saves files correctly"""
    # Test without saving
    plot_class_distributions_comparison(sample_distributions)
    mock_show.assert_called_once()

    # Test with saving
    save_path = temp_path / "class_dist_comparison.png"
    os.makedirs(temp_path, exist_ok=True)
    plot_class_distributions_comparison(sample_distributions, save_path=str(save_path))
    assert save_path.exists()


@patch("matplotlib.pyplot.show")
def test_plot_comparison_results(mock_show, sample_results, temp_path):
    """Test if plot_comparison_results runs without errors and saves files correctly"""
    # Test without saving
    plot_comparison_results(sample_results)
    mock_show.assert_called_once()

    # Test with saving
    save_path = temp_path / "comparison_results.png"
    os.makedirs(temp_path, exist_ok=True)
    plot_comparison_results(sample_results, save_path=str(save_path))
    assert save_path.exists()


@patch("matplotlib.pyplot.show")
def test_plot_learning_curves(mock_show, sample_learning_curve_data, temp_path):
    """Test if plot_learning_curves runs without errors and saves files correctly"""
    # Test without saving
    plot_learning_curves(sample_learning_curve_data)
    mock_show.assert_called_once()

    # Test with saving
    save_path = temp_path / "learning_curves.png"
    os.makedirs(temp_path, exist_ok=True)
    plot_learning_curves(sample_learning_curve_data, save_path=str(save_path))
    assert save_path.exists()


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
        plot_comparison_results({})

    with pytest.raises(TypeError):
        plot_comparison_results(None)


def test_plot_learning_curves_invalid_input():
    """Test if plot_learning_curves handles invalid input correctly"""
    with pytest.raises((ValueError, TypeError)):
        plot_learning_curves({})

    with pytest.raises(TypeError):
        plot_learning_curves(None)
