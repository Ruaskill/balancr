import time
from typing import Dict, List, Optional, Union, Any
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from .technique_registry import TechniqueRegistry
from .data import DataLoader
from .classifier_registry import ClassifierRegistry
from .data import DataPreprocessor
from .evaluation import (
    get_metrics,
    get_cv_scores,
    get_learning_curve_data_multiple_techniques,
    get_learning_curve_data_against_imbalanced_multiple_techniques,
)
from .evaluation import (
    plot_class_distribution,
    plot_class_distributions_comparison,
    plot_comparison_results,
    plot_learning_curves,
)


def format_time(seconds):
    """Format time in seconds to minutes and seconds"""
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}mins, {remaining_seconds:.2f}secs"


class BalancingFramework:
    """
    A unified framework for analysing and comparing different techniques
    for handling imbalanced data.
    """

    def __init__(self):
        """Initialise the framework with core components."""
        self.technique_registry = TechniqueRegistry()
        self.preprocessor = DataPreprocessor()
        self.classifier_registry = ClassifierRegistry()
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.results = {}
        self.current_data_info = {}
        self.current_balanced_datasets = {}
        self.quality_report = {}

    def load_data(
        self,
        file_path: Union[str, Path],
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        auto_preprocess: bool = False,
        correlation_threshold: float = 0.95,
    ) -> None:
        """
        Load data from a file and optionally preprocess it.

        Args:
            file_path: Path to the data file
            target_column: Name of the target column
            feature_columns: List of feature columns to use (optional)
            auto_preprocess: Whether to automatically preprocess the data
        """
        # Load data
        self.X, self.y = DataLoader.load_data(file_path, target_column, feature_columns)

        if feature_columns is None:
            # Need to re-determine what columns were actually used
            import pandas as pd

            data = pd.read_csv(file_path)  # Re-read the data
            feature_columns = [col for col in data.columns if col != target_column]

        # Store data info
        self.current_data_info = {
            "file_path": file_path,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "original_shape": self.X.shape,
            "class_distribution": self._get_class_distribution(),
        }

        # Check data quality
        quality_report = self.preprocessor.check_data_quality(
            self.X, feature_columns, correlation_threshold=correlation_threshold
        )

        self.quality_report = quality_report
        self._handle_quality_issues(
            quality_report, correlation_threshold=correlation_threshold
        )

        if auto_preprocess:
            self.preprocess_data()

    def preprocess_data(
        self,
        handle_missing: str = "mean",
        scale: str = "standard",
        handle_constant_features: str = "none",
        handle_correlations: str = "none",
        categorical_features: Optional[List[str]] = None,
        hash_components_dict: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Preprocess the loaded data with enhanced options.

        Args:
            handle_missing: Strategy to handle missing values
                ("drop", "mean", "median", "mode", "none")
            scale: Scaling method
                ("standard", "minmax", "robust", "none")
            handle_constant_features: Strategy to handle constant features
                ("drop", "none")
            handle_correlations: Strategy to handle highly correlated features
                ("drop_first", "drop_lowest", "pca", "none")
            categorical_features: List of column names for categorical features
            hash_components_dict: Dictionary mapping feature names to number of hash components
        """
        if self.X is None or self.y is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Extract constant features and correlated features from quality report
        constant_features = []
        if self.quality_report and "constant_features" in self.quality_report:
            # Extract feature names from the constant_features list of tuples
            constant_features = [
                feature[0]
                for feature in self.quality_report.get("constant_features", [])
            ]

        correlated_features = []
        if self.quality_report and "feature_correlations" in self.quality_report:
            # Extract correlation pairs from the feature_correlations list of tuples
            correlated_features = self.quality_report.get("feature_correlations", [])

        # Process the data using the preprocessor with all options
        self.X, self.y = self.preprocessor.preprocess(
            self.X,
            self.y,
            handle_missing=handle_missing,
            scale=scale,
            handle_constant_features=handle_constant_features,
            handle_correlations=handle_correlations,
            constant_features=constant_features,
            correlated_features=correlated_features,
            all_features=self.current_data_info.get("feature_columns"),
            categorical_features=categorical_features,
            hash_components_dict=hash_components_dict,
        )

        # Update feature columns in current_data_info with the new feature names
        if hasattr(self.preprocessor, "feature_names"):
            self.current_data_info["feature_columns"] = self.preprocessor.feature_names

    def inspect_class_distribution(
        self, save_path: Optional[str] = None, display: bool = False
    ) -> Dict[Any, int]:
        """
        Inspect the distribution of classes in the target variable.

        Args:
            plot: Whether to create a visualisation

        Returns:
            Dictionary mapping class labels to their counts
        """
        if self.y is None:
            raise ValueError("No data loaded. Call load_data() first.")

        distribution = self._get_class_distribution()

        plot_class_distribution(
            distribution,
            title="Imbalanced Dataset Class Comparison",
            save_path=save_path,
            display=display,
        )

        return distribution

    def list_available_techniques(self) -> Dict[str, List[str]]:
        """List all available balancing techniques."""
        return self.technique_registry.list_available_techniques()

    def apply_balancing_techniques(
        self,
        technique_names: List[str],
        test_size: float = 0.2,
        random_state: int = 42,
        technique_params: Optional[Dict[str, Dict[str, Any]]] = None,
        include_original: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Apply multiple balancing techniques to the dataset.

        Args:
            technique_names: List of technique names to apply
            test_size: Proportion of dataset to use for testing
            random_state: Random seed for reproducibility
            technique_params: Dictionary mapping technique names to their parameters

        Returns:
            Dictionary containing balanced datasets for each technique
        """
        if self.X is None or self.y is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        # Store test data for later evaluation
        self.X_test = X_test
        self.y_test = y_test

        balanced_datasets = {}

        if include_original:
            # Store imbalanced dataset to compare with balanced later
            balanced_datasets["Original"] = {
                "X_balanced": X_train,
                "y_balanced": y_train,
            }

        for technique_name in technique_names:
            # Get technique
            technique_class = self.technique_registry.get_technique_class(
                technique_name
            )
            if technique_class is None:
                raise ValueError(
                    f"Technique '{technique_name}' not found. "
                    f"Available techniques: {self.list_available_techniques()}"
                )

            # Get parameters for this technique
            params = {}
            if technique_params and technique_name in technique_params:
                params = technique_params[technique_name]

            # Apply technique with parameters
            technique = technique_class(**params)
            X_balanced, y_balanced = technique.balance(X_train, y_train)

            # Store balanced data
            balanced_datasets[technique_name] = {
                "X_balanced": X_balanced,
                "y_balanced": y_balanced,
            }

        # Update current balanced datasets
        self.current_balanced_datasets = balanced_datasets

        return balanced_datasets

    def train_classifiers(
        self,
        classifier_configs: Dict[str, Dict[str, Any]] = None,
        enable_cv: bool = False,
        cv_folds: int = 5,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Train classifiers on balanced datasets and evaluate their performance.

        Args:
            classifier_configs: Dictionary mapping classifier names to their parameters
                               If None, uses default RandomForestClassifier
            plot_results: Whether to visualise the comparison results
            enable_cv: Whether to perform cross-validation evaluation
            cv_folds: Number of cross-validation folds (if enabled)

        Returns:
            Dictionary mapping classifier names to technique results
        """
        if not self.current_balanced_datasets:
            raise ValueError(
                "No balanced datasets available. Run apply_balancing_techniques first."
            )

        if self.X_test is None or self.y_test is None:
            raise ValueError(
                "Test data not found. Run apply_balancing_techniques first."
            )

        # Default classifier if none provided
        if classifier_configs is None:
            classifier_configs = {"RandomForestClassifier": {"random_state": 42}}

        # Initialise results dictionary
        results = {}

        # For each classifier
        for clf_name, clf_params in classifier_configs.items():
            # Get classifier class from registry
            clf_class = self.classifier_registry.get_classifier_class(clf_name)

            if clf_class is None:
                logging.warning(
                    f"Classifier '{clf_name}' not found in registry. Skipping."
                )
                continue

            classifier_results = {}

            # For each balancing technique
            for technique_name, balanced_data in self.current_balanced_datasets.items():
                X_balanced = balanced_data["X_balanced"]
                y_balanced = balanced_data["y_balanced"]

                try:
                    # Create classifier instance with parameters
                    clf_instance = clf_class(**clf_params)

                    # Train the classifier
                    start_time = time.time()
                    logging.info(f"Training {clf_name} with dataset balanced with {technique_name}...")
                    clf_instance.fit(X_balanced, y_balanced)
                    train_time = time.time() - start_time
                    logging.info(f"Training {clf_name} with dataset balanced with {technique_name} complete"
                                 f"(Time Taken: {format_time(train_time)})")

                    # Initialise metrics for this technique
                    start_time = time.time()
                    logging.info(f"Getting standard metrics of {technique_name} after training {clf_name}...")
                    technique_metrics = {
                        "standard_metrics": get_metrics(
                            clf_instance,
                            self.X_test,
                            self.y_test,
                        )
                    }
                    std_metrics_time = time.time() - start_time
                    logging.info(f"Getting standard metrics of {technique_name} after training {clf_name} complete"
                                 f"(Time Taken: {format_time(std_metrics_time)})")

                    # Add cross-validation metrics if enabled
                    if enable_cv:
                        start_time = time.time()
                        logging.info(f"Getting cv metrics of {technique_name} after training {clf_name}...")
                        technique_metrics["cv_metrics"] = get_cv_scores(
                            clf_class(**clf_params),
                            X_balanced,
                            y_balanced,
                            n_folds=cv_folds,
                        )
                        cv_metrics_time = time.time() - start_time
                        logging.info(f"Getting cv metrics of {technique_name} after training {clf_name} complete"
                                     f"(Time Taken: {format_time(cv_metrics_time)})")

                    classifier_results[technique_name] = technique_metrics

                except Exception as e:
                    logging.error(
                        f"Error training classifier '{clf_name}' with technique '{technique_name}': {str(e)}"
                    )
                    continue

            # Only add classifier results if at least one technique was successful
            if classifier_results:
                results[clf_name] = classifier_results

        # Update the overall results
        self.results = results

        return results

    def save_results(
        self,
        file_path: Union[str, Path],
        file_type: str = "csv",
        include_plots: bool = True,
    ) -> None:
        """
        Save comparison results to a file.

        Args:
            file_path: Path to save the results
            file_type: Type of file ('csv' or 'json')
            include_plots: Whether to save visualisation plots
        """
        if not self.results:
            raise ValueError("No results to save. Run compare_techniques() first.")

        file_path = Path(file_path)

        # Save results
        if file_type == "csv":
            pd.DataFrame(self.results).to_csv(file_path)
        elif file_type == "json":
            pd.DataFrame(self.results).to_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Save plots if requested
        if include_plots:
            plot_path = file_path.parent / f"{file_path.stem}_plots.png"
            plot_comparison_results(self.results, save_path=plot_path)

    def save_classifier_results(
        self,
        file_path: Union[str, Path],
        classifier_name: str,
        metric_type: str = "standard_metrics",
        file_type: str = "csv",
    ) -> None:
        """
        Save results for a specific classifier and metric type to a file.

        Args:
            file_path: Path to save the results
            classifier_name: Name of the classifier to extract results for
            metric_type: Type of metrics to save ('standard_metrics' or 'cv_metrics')
            file_type: Type of file ('csv' or 'json')
        """
        if not self.results:
            raise ValueError("No results to save. Run train_classifiers() first.")

        if classifier_name not in self.results:
            raise ValueError(f"Classifier '{classifier_name}' not found in results.")

        file_path = Path(file_path)

        # Extract results for this classifier
        classifier_results = self.results[classifier_name]

        # Create a dictionary where keys are techniques and values are the metrics
        extracted_results = {}
        for technique_name, technique_data in classifier_results.items():
            if metric_type in technique_data:
                extracted_results[technique_name] = technique_data[metric_type]

        # Convert to DataFrame for easier saving
        results_df = pd.DataFrame(extracted_results)

        # Save results in requested format
        if file_type == "csv":
            results_df.to_csv(file_path)
        elif file_type == "json":
            results_df.to_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        logging.info(f"Saved {classifier_name} {metric_type} results to {file_path}")

    def generate_balanced_data(
        self,
        folder_path: str,
        techniques: Optional[List[str]] = None,
        file_format: str = "csv",
    ) -> None:
        """
        Save balanced datasets to files for specified techniques.

        Args:
            folder_path: Directory to save the datasets.
            techniques: List of techniques to save. Saves all if None.
            file_format: Format for saving the data ('csv' or 'json').

        Raises:
            ValueError if no balanced datasets are available or specified techniques are invalid.
        """

        # Validate datasets exist
        if not self.current_balanced_datasets:
            raise ValueError(
                "No balanced datasets available. Run compare_techniques first."
            )

        # Validate output format
        if file_format not in ["csv", "json"]:
            raise ValueError("Invalid file format. Supported formats: 'csv', 'json'.")

        # Create output folder
        os.makedirs(folder_path, exist_ok=True)

        # Determine techniques to save
        if techniques is None:
            techniques = list(self.current_balanced_datasets.keys())

        # Retrieve input data column names
        feature_columns = self.current_data_info.get("feature_columns")
        target_column = self.current_data_info.get("target_column")
        if feature_columns is None or target_column is None:
            raise ValueError(
                "Original column names are missing in 'current_data_info'."
            )

        # Export datasets
        for technique in techniques:
            if technique not in self.current_balanced_datasets:
                raise ValueError(
                    f"Technique '{technique}' not found in current datasets."
                )

            # Retrieve data
            dataset = self.current_balanced_datasets[technique]
            X_balanced = dataset["X_balanced"]
            y_balanced = dataset["y_balanced"]

            # Combine into a single DataFrame
            balanced_df = pd.DataFrame(X_balanced, columns=feature_columns)
            balanced_df[target_column] = y_balanced

            # Construct file path
            file_path = os.path.join(folder_path, f"balanced_{technique}.{file_format}")

            # Save in the chosen format
            if file_format == "csv":
                balanced_df.to_csv(file_path, index=False)
            elif file_format == "json":
                balanced_df.to_json(file_path, index=False)

            logging.info(f"Saved balanced dataset for '{technique}' to {file_path}")

    def compare_balanced_class_distributions(
        self, save_path: Optional[str] = None, display: bool = False
    ) -> None:
        """
        Compare class distributions of balanced datasets for all techniques.

        Args:
            save_path: Path to save the visualisation (optional).

        Raises:
            ValueError: If no balanced datasets are available.
        """
        if not self.current_balanced_datasets:
            raise ValueError(
                "No balanced datasets available. Run compare_techniques first."
            )

        # Generate class distributions for each balanced dataset
        distributions = {}
        for technique, dataset in self.current_balanced_datasets.items():
            y_balanced = dataset["y_balanced"]

            # Generate class distribution
            distribution = self.preprocessor.inspect_class_distribution(y_balanced)
            distributions[technique] = distribution

        # Call the visualisation function
        plot_class_distributions_comparison(
            distributions,
            title="Class Distribution Comparison After Balancing",
            save_path=save_path,
            display=display,
        )

    def generate_learning_curves(
        self,
        classifier_name: str,
        learning_curve_type: str,
        train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
        n_folds: int = 5,
        save_path: Optional[str] = None,
        display: bool = False,
    ) -> None:
        """
        Generate and plot learning curves for multiple balancing techniques.

        Args:
            classifier_name: Name of the classifier to generate curves for
            train_sizes: Training set sizes to evaluate
            n_folds: Number of cross-validation folds
            save_path: Path to save the plot (optional)
            display: Whether to display the plot
        """
        if not self.current_balanced_datasets:
            raise ValueError(
                "No balanced datasets available. Run apply_balancing_techniques first."
            )

        try:
            # Get the classifier class
            clf_class = self.classifier_registry.get_classifier_class(classifier_name)
            if clf_class is None:
                logging.warning(
                    f"Classifier '{classifier_name}' not found. Skipping learning curves."
                )
                return

            # Get classifier parameters from configuration
            clf_params = {}
            if (
                hasattr(self, "classifier_configs")
                and classifier_name in self.classifier_configs
            ):
                clf_params = self.classifier_configs[classifier_name]

            # Create classifier instance with the same parameters used in training
            classifier = clf_class(**clf_params)

            if learning_curve_type == "Balanced Datasets":
                learning_curve_data = get_learning_curve_data_multiple_techniques(
                    classifier_name=classifier_name,
                    classifier=classifier,
                    techniques_data=self.current_balanced_datasets,
                    train_sizes=train_sizes,
                    n_folds=n_folds,
                )

                title = f"Learning Curves for {classifier_name} Evaluated Against {learning_curve_type}"

                plot_learning_curves(
                    learning_curve_data,
                    title=title,
                    save_path=save_path,
                    display=display
                )
            elif learning_curve_type == "Original Dataset":
                learning_curve_data = get_learning_curve_data_against_imbalanced_multiple_techniques(
                    classifier_name=classifier_name,
                    classifier=classifier,
                    techniques_data=self.current_balanced_datasets,
                    X_test=self.X_test,
                    y_test=self.y_test,
                    train_sizes=train_sizes,
                    n_folds=n_folds,
                )

                title = f"Learning Curves for {classifier_name} Evaluated Against {learning_curve_type}"

                plot_learning_curves(
                    learning_curve_data, title=title, save_path=save_path, display=display
                )

        except Exception as e:
            logging.warning(
                f"Failed to generate learning curves for classifier '{classifier_name}': {str(e)}"
            )
            logging.warning("Continuing with other visualisations...")

    def _get_class_distribution(self) -> Dict[Any, int]:
        """Get the distribution of classes in the target variable."""
        return self.preprocessor.inspect_class_distribution(self.y)

    def _handle_quality_issues(
        self, quality_report: Dict[str, Any], correlation_threshold: float = 0.95
    ) -> None:
        """Handle any data quality issues found."""
        warnings = []

        # Check if there are any missing values (now a list of tuples)
        if quality_report["missing_values"]:
            missing_value_info = ", ".join(
                [f"{name}: {count}" for name, count in quality_report["missing_values"]]
            )
            if len(quality_report["missing_values"]) == 1:
                warnings.append(
                    f"Data contains missing values in feature: {missing_value_info}"
                )
            else:
                warnings.append(
                    f"Data contains missing values in features: {missing_value_info}"
                )

        # Check if there are any constant features (now a list of tuples)
        if quality_report["constant_features"]:
            # Extract feature names from the tuples
            constant_feature_names = [
                name for name, _ in quality_report["constant_features"]
            ]
            if len(constant_feature_names) == 1:
                warnings.append(
                    f"Constant Features: {constant_feature_names} has constant values"
                )
            else:
                warnings.append(
                    f"Constant Features: {constant_feature_names} have constant values"
                )

        # Check if there are any highly correlated features (already a list)
        if quality_report["feature_correlations"]:
            # Format the correlation information more readably
            correlation_info = ", ".join(
                [
                    f"{col1} & {col2} ({corr:.2f})"
                    for col1, col2, corr in quality_report["feature_correlations"]
                ]
            )
            warnings.append(
                f"Found highly correlated features (Threshold={correlation_threshold}): {correlation_info}"
            )

        # Print all warnings
        if warnings:
            print("Data Quality Warnings:")
            for warning in warnings:
                print(f"- {warning}")
