# Balancr: A Unified Framework for Analysing Data Balancing Techniques

A comprehensive framework and CLI tool for analysing and comparing different techniques for handling imbalanced datasets in machine learning. Balancr makes it easier to compare balancing algorithims against a wide range of classifers 

## Overview

Imbalanced datasets are a significant challenge in machine learning, particularly in areas such as:
- Medical diagnosis
- Fraud detection
- Network intrusion detection
- Rare event prediction

Balancr allows you to:
- Compare different balancing techniques (e.g., SMOTE, ADASYN, random undersampling), and the same technqiues with different configurations, against multiple classifiers
- Evaluate performance using relevant metrics
- Visualise results and class distributions
- Generate balanced datasets using various methods
- Customise the evaluation process with different classifiers

## Features

### Core Functionality:
- **CLI Interface**: Simple command-line interface for full workflow
- **Data Loading**: Support for CSV, and provides a data quality check
- **Preprocessing**: Configurable preprocessing functionality including handling data quality issues, scaling, and encoding categorical features 
- **Dynamic Technique Discovery**: Automatic discovery of techniques from imbalanced-learn
- **Custom Technique Registration**: Register your own balancing techniques
- **Classifier Selection**: Compare performance across multiple classifiers.
Fine-tune parameters via the configuration file
- **Custom Classifier Registration**: Register your own classifier implementations
- **Comprehensive Metric Evaluation**: Get metrics specific to imbalanced learning
- **Visualisation Suite**: Plots for class distributions, metrics comparison, and learning curves (more to come)
- **Flexible Configuration**: Configure every aspect via CLI or configuration file

### Available Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC
- G-mean
- Specificity
- Cross-validation scores

### Visualisations
- Class distribution comparisons
- Performance metric comparisons
- Learning curves
- Results comparison plots

## Installation

```bash
# From PyPI (recommended)
pip install balancr

# From source
git clone https://gitlab.eeecs.qub.ac.uk/40353634/csc3002-balancing-techniques-framework.git
cd balancing-techniques-framework
pip install -e .
```

## Command-Line Interface

Balancr provides a comprehensive CLI to help you analyse imbalanced datasets:

```
  ____        _                       
 | __ )  __ _| | __ _ _ __   ___ _ __ 
 |  _ \ / _` | |/ _` | '_ \ / __| '__|
 | |_) | (_| | | (_| | | | | (__| |   
 |____/ \__,_|_|\__,_|_| |_|\\___|_|   
                                     
```

### Quick Start - CLI

Here's a complete workflow using the CLI:

```bash
# Load your dataset
balancr load-data dataset.csv -t target_column

# Configure preprocessing
balancr preprocess --scale standard --handle-missing mean --encode auto

# Select balancing techniques to compare
balancr select-techniques SMOTE RandomUnderSampler ADASYN

# Select classifiers for evaluation
balancr select-classifiers RandomForestClassifier LogisticRegression

# Configure metrics
balancr configure-metrics --metrics precision recall f1 roc_auc

# Configure visualisations
balancr configure-visualisations --types all --save-formats png

# Configure evaluation settings
balancr configure-evaluation --test-size 0.3 --cross-validation 5

# Run the comparison
balancr run --output-dir results/experiment1
```

### Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `load-data` | Load a dataset for analysis | `balancr load-data dataset.csv -t target` |
| `preprocess` | Configure preprocessing options | `balancr preprocess --scale standard --handle-missing mean` |
| `select-techniques` | Select balancing techniques | `balancr select-techniques SMOTE ADASYN` |
| `register-techniques` | Register custom techniques | `balancr register-techniques my_technique.py` |
| `select-classifiers` | Select classifiers for evaluation | `balancr select-classifiers RandomForestClassifier` |
| `register-classifiers` | Register custom classifiers | `balancr register-classifiers my_classifier.py` |
| `configure-metrics` | Configure evaluation metrics | `balancr configure-metrics --metrics precision recall f1` |
| `configure-visualisations` | Configure visualisation options | `balancr configure-visualisations --types all` |
| `configure-evaluation` | Configure model evaluation settings | `balancr configure-evaluation --test-size 0.3` |
| `run` | Run comparison of techniques | `balancr run --output-dir results` |
| `reset` | Reset configuration to defaults | `balancr reset` |

## Python API

Balancr can also be used as a Python library:

```python
from balancr.imbalance_analyser import BalancingFramework

# Initialize the framework
framework = BalancingFramework()

# Load your dataset
framework.load_data(
    file_path="path/to/your/data.csv",
    target_column="target",
    feature_columns=["feature1", "feature2", "feature3"]
)

# Preprocess the data
framework.preprocess_data(
    handle_missing="mean",
    scale="standard",
    encode="auto"
)

# Apply balancing techniques
balanced_datasets = framework.apply_balancing_techniques(
    technique_names=["SMOTE", "RandomUnderSampler", "ADASYN"],
    test_size=0.2
)

# Train and evaluate classifiers
results = framework.train_classifiers(
    classifier_configs={
        "RandomForestClassifier": {"n_estimators": 100, "random_state": 42},
        "LogisticRegression": {"C": 1.0, "random_state": 42}
    },
    enable_cv=True,
    cv_folds=5
)

# Generate visualisations
framework.compare_balanced_class_distributions(
    save_path="results/class_distributions.png"
)

# Generate learning curves
framework.generate_learning_curves(
    classifier_name="RandomForestClassifier",
    save_path="results/learning_curves.png"
)

# Save results
framework.save_classifier_results(
    "results/metrics_results.csv",
    classifier_name="RandomForestClassifier"
)
```

## Creating Custom Techniques

You can create and register your own balancing techniques:

```python
from balancr.base import BaseBalancer
import numpy as np

class MyCustomBalancer(BaseBalancer):
    """A custom balancing technique that implements your logic"""
    
    def balance(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Implement your balancing logic here
        # This should return the balanced X and y
        
        return X_balanced, y_balanced
```

Register your technique using the CLI:

```bash
balancr register-techniques my_custom_technique.py
```

Or using the Python API:

```python
from balancr.technique_registry import TechniqueRegistry
from my_custom_technique import MyCustomBalancer

registry = TechniqueRegistry()
registry.register_custom_technique("MyCustomBalancer", MyCustomBalancer)
```

## Creating Custom Classifiers

You can create and register your own classifiers:

```python
from sklearn.base import BaseEstimator
import numpy as np

class CustomClassifier(BaseEstimator):
    def __init__(self, n_estimators, random_state):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        # Implement your training logic here
        # Return self. Fitted estimator.
        return self

    def predict(self, X):
        # Implement your prediction logic here
        # Return your predictions/list of predicitons
        return np.zeros(len(X))
```

Register your classifier using the CLI:

```bash
balancr register-classifier my_custom_classifier.py
```

Or using the Python API:

```python
from balancr.classifier_registry import ClassifierRegistry
from my_custom_classifier import MyCustomClassifier

registry = ClassifierRegistry()
registry.register_custom_classifier("MyCustomClassifier", MyCustomClassifier)
```

## Extra Configuration Tips

### Manual Configurations
For more control, all confiurgations are stored in balancr's config file (default location: ~/.balancr/config.json)

### Comparing Balancers/Classifiers Against Themselves

To be able to compare a balancing technique or classifier against itself, but with different parameters, as is often the case, extra configuration is required

### To compare a balancing technique against itself:
First, select the balancer you want to compare:

```bash
balancr select-techniques SMOTE
```

Then in balancr's config file (default location: ~/.balancr/config.json), you should see the config settings of your selected technique:

```json
  "balancing_techniques": {
    "SMOTE": {
      "sampling_strategy": "auto",
      "random_state": 42,
      "k_neighbors": 3,
      "n_jobs": null
    }
  },
```

Create a copy of this technique config, and make sure to change the name to contain a valid suffix (suffix needs to start with _ or -), e.g. SMOTE_v2, SMOTE-2, SMOTE_ChangedParams, etc.:

```json
  "balancing_techniques": {
    "SMOTE": {
      "sampling_strategy": "auto",
      "random_state": 42,
      "k_neighbors": 3,
      "n_jobs": null
    },
    "SMOTE_v2": {
      "sampling_strategy": "auto",
      "random_state": 42,
      "k_neighbors": 3,
      "n_jobs": null
    }
  },
```

You can then change the desired parameters. In this exmaple, we change k_neighbors from 3 to 5:
```json
  "balancing_techniques": {
    "SMOTE": {
      "sampling_strategy": "auto",
      "random_state": 42,
      "k_neighbors": 3,
      "n_jobs": null
    },
    "SMOTE_v2": {
      "sampling_strategy": "auto",
      "random_state": 42,
      "k_neighbors": 5, 
      "n_jobs": null
    }
  },
```
### To compare a classifier against itself:
First, select the classifier you want to compare:

```bash
balancr select-classifiers RandomForestClassifier
```

Then in balancr's config file (default location: ~/.balancr/config.json), you should see the config settings of your selected classifier:

```json
    "classifiers": {
    "RandomForestClassifier": {
      "n_estimators": 100,
      "criterion": "gini",
      "max_depth": null,
      "min_samples_split": 2,
      "min_samples_leaf": 1,
      "min_weight_fraction_leaf": 0.0,
      "max_features": "sqrt",
      "max_leaf_nodes": null,
      "min_impurity_decrease": 0.0,
      "bootstrap": true,
      "oob_score": false,
      "n_jobs": null,
      "random_state": null,
      "verbose": 0,
      "warm_start": false,
      "class_weight": null,
      "ccp_alpha": 0.0,
      "max_samples": null,
      "monotonic_cst": null
    }
  },
```

Create a copy of this classifier config, and make sure to change the name to contain a valid suffix (suffix needs to start with _ or -), e.g. RandomForestClassifier_v2, RandomForestClassifier-2, RandomForestClassifier_ChangedParams, etc.:

```json
    "classifiers": {
    "RandomForestClassifier": {
      "n_estimators": 100,
      "criterion": "gini",
      "max_depth": null,
      "min_samples_split": 2,
      "min_samples_leaf": 1,
      "min_weight_fraction_leaf": 0.0,
      "max_features": "sqrt",
      "max_leaf_nodes": null,
      "min_impurity_decrease": 0.0,
      "bootstrap": true,
      "oob_score": false,
      "n_jobs": null,
      "random_state": null,
      "verbose": 0,
      "warm_start": false,
      "class_weight": null,
      "ccp_alpha": 0.0,
      "max_samples": null,
      "monotonic_cst": null
    },
    "RandomForestClassifier_v2": {
      "n_estimators": 100,
      "criterion": "gini",
      "max_depth": null,
      "min_samples_split": 2,
      "min_samples_leaf": 1,
      "min_weight_fraction_leaf": 0.0,
      "max_features": "sqrt",
      "max_leaf_nodes": null,
      "min_impurity_decrease": 0.0,
      "bootstrap": true,
      "oob_score": false,
      "n_jobs": null,
      "random_state": null,
      "verbose": 0,
      "warm_start": false,
      "class_weight": null,
      "ccp_alpha": 0.0,
      "max_samples": null,
      "monotonic_cst": null
    }
  },
```

You can then change the desired parameters. In this exmaple, we change n_estimators from 100 to 200:
```json
    "classifiers": {
    "RandomForestClassifier": {
      "n_estimators": 100,
      "criterion": "gini",
      "max_depth": null,
      "min_samples_split": 2,
      "min_samples_leaf": 1,
      "min_weight_fraction_leaf": 0.0,
      "max_features": "sqrt",
      "max_leaf_nodes": null,
      "min_impurity_decrease": 0.0,
      "bootstrap": true,
      "oob_score": false,
      "n_jobs": null,
      "random_state": null,
      "verbose": 0,
      "warm_start": false,
      "class_weight": null,
      "ccp_alpha": 0.0,
      "max_samples": null,
      "monotonic_cst": null
    },
    "RandomForestClassifier_v2": {
      "n_estimators": 200,
      "criterion": "gini",
      "max_depth": null,
      "min_samples_split": 2,
      "min_samples_leaf": 1,
      "min_weight_fraction_leaf": 0.0,
      "max_features": "sqrt",
      "max_leaf_nodes": null,
      "min_impurity_decrease": 0.0,
      "bootstrap": true,
      "oob_score": false,
      "n_jobs": null,
      "random_state": null,
      "verbose": 0,
      "warm_start": false,
      "class_weight": null,
      "ccp_alpha": 0.0,
      "max_samples": null,
      "monotonic_cst": null
    }
  },
```

### Cross Validation
Cross validation can be enabled to be applied to balanced training data only. This gives an estimate of how well a classifier can learn from the balanced data and generalise across different parts of that balanced dataset. 

To apply cross validation with balanced datasets, apply a cross validation number with:
```bash
balancr configure-evaluation 5
```
This will perform cross validation with 5 folds, meaning each balanced dataset will be split into 5 folds, and will train the selected classifiers in 5 rounds.

An average of these round's results will be retrieved

### Learning Curves
The same process mentioned in cross validation above is applied to generating learning curves.

Learning curves help us visualise each model's performance when being trained on increasing amounts of data.

These learning curves are generated using the balanced datasets chosen by the user

To configure learning curves:
```bash
balancr configure-evaluations --learning-curve-folds 5 --learning-curve-points 10
```
This will set the number of cross validation folds and number of points to plot on the learning curves

## Requirements

- Python >= 3.8
- NumPy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- imbalanced-learn >= 0.8.0
- openpyxl >= 3.0.0
- colorama >= 0.4.4

## Future Plans

- More visualisation options
- Collecting balancer and classifier times, other than only displaying in logs
- Saving results as runs go along, rather than retrieving all results at end of run

## Author

Conor Doherty, cdoherty135@qub.ac.uk