# Balancing Techniques Analysis Framework

A unified framework for analysing and comparing different techniques for handling imbalanced datasets in machine learning. This framework provides a standardised way to evaluate various data balancing methods, making it easier to determine the most effective approach for specific imbalanced data scenarios.

## Overview

Imbalanced datasets are a significant challenge in machine learning, particularly in areas such as:
- Medical diagnosis
- Fraud detection
- Network intrusion detection
- Rare event prediction

This framework allows users to:
- Compare different balancing techniques (e.g., SMOTE, ADASYN, random undersampling)
- Evaluate performance using relevant metrics
- Visualise results and class distributions
- Generate balanced datasets using various methods

## Current State

This project is currently available as a cloneable repository and is under active development. Future releases will include packaging and distribution through PyPI.

## Requirements (all in requirements.txt)

- Python >= 3.8
- NumPy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- imbalanced-learn >= 0.8.0

## Installation

Currently, the framework can be installed by cloning this repository:

```bash
git clone https://github.com/yourusername/balancing-techniques-framework.git
cd balancing-techniques-framework
pip install -r requirements.txt
```

## Quick Start

Here's a basic example of how to use the framework:

```python
from imbalance_framework.imbalance_analyser import BalancingFramework

# Initialize the framework
framework = BalancingFramework()

# Load your dataset
framework.load_data(
    file_path="path/to/your/data.csv",
    target_column="target",
    feature_columns=["feature1", "feature2", "feature3"]
)

# Inspect class distribution
distribution = framework.inspect_class_distribution()

# Compare different balancing techniques
results = framework.compare_techniques(
    technique_names=["SMOTE", "RandomUnderSampler", "SVMSMOTE"],
    test_size=0.2
)

# Save results and plots
framework.save_results(
    "results/comparison_results.csv",
    include_plots=True
)

# Generate balanced versions of your dataset
framework.generate_balanced_data("results/balanced/")
```

## Features

### Core Functionality
- Data loading and preprocessing
- Automatic technique discovery from imbalanced-learn
- Custom technique registration
- Comprehensive metric evaluation
- Results visualisation

### Available Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC
- G-mean
- Specificity

### Visualisations
- Class distribution comparisons
- Performance metric comparisons
- Learning curves
- Results comparison plots

## Future Plans

- Package distribution through PyPI
- Enhanced visualisation options
- Dynamic classifer selection

## Author

Conor Doherty, cdoherty135@qub.ac.uk
