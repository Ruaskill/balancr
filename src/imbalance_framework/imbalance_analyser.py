from typing import Dict, List, Optional, Union, Any
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from .technique_registry import TechniqueRegistry
from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from evaluation.metrics import (
    get_metrics,
    get_learning_curve_data_multiple_techniques,
)
from evaluation.visualisation import (
    plot_class_distribution,
    plot_class_distributions_comparison,
    plot_comparison_results,
    plot_learning_curves,
)


class BalancingFramework:
    """
    A unified framework for analyzing and comparing different techniques
    for handling imbalanced data.
    """

    def __init__(self):
        """Initialize the framework with core components."""
        self.registry = TechniqueRegistry()
        self.preprocessor = DataPreprocessor()
        self.X = None
        self.y = None
        self.results = {}
        self.current_data_info = {}
        self.current_balanced_datasets = {}

    def load_data(
        self,
        file_path: Union[str, Path],
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        auto_preprocess: bool = True,
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

        # Store data info
        self.current_data_info = {
            "file_path": file_path,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "original_shape": self.X.shape,
            "class_distribution": self._get_class_distribution(),
        }

        # Check data quality
        quality_report = self.preprocessor.check_data_quality(self.X)
        self._handle_quality_issues(quality_report)

        if auto_preprocess:
            self.preprocess_data()

    def preprocess_data(
        self,
        handle_missing: str = "mean",
        scale: str = "standard",
        encode: str = "auto",
    ) -> None:
        """
        Preprocess the loaded data with enhanced options.

        Args:
            handle_missing: Strategy to handle missing values
                ("drop", "mean", "median", "mode", "none")
            scale: Scaling method
                ("standard", "minmax", "robust", "none")
            encode: Encoding method for categorical features
                ("auto", "onehot", "label", "ordinal", "none")
        """
        if self.X is None or self.y is None:
            raise ValueError("No data loaded. Call load_data() first.")

        self.X, self.y = self.preprocessor.preprocess(
            self.X, self.y, handle_missing=handle_missing, scale=scale, encode=encode
        )

    def inspect_class_distribution(self, plot: bool = True) -> Dict[Any, int]:
        """
        Inspect the distribution of classes in the target variable.

        Args:
            plot: Whether to create a visualization

        Returns:
            Dictionary mapping class labels to their counts
        """
        if self.y is None:
            raise ValueError("No data loaded. Call load_data() first.")

        distribution = self._get_class_distribution()

        if plot:
            plot_class_distribution(distribution)

        return distribution

    def list_available_techniques(self) -> Dict[str, List[str]]:
        """List all available balancing techniques."""
        return self.registry.list_available_techniques()

    def compare_techniques(
        self,
        technique_names: List[str],
        test_size: float = 0.2,
        random_state: int = 42,
        plot_results: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple balancing techniques using various metrics.

        Args:
            technique_names: List of technique names to compare
            test_size: Proportion of dataset to use for testing
            random_state: Random seed for reproducibility
            plot_results: Whether to visualize the comparison results

        Returns:
            Dictionary containing results for each technique
        """
        if self.X is None or self.y is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        results = {}
        for technique_name in technique_names:
            # Get technique
            technique_class = self.registry.get_technique_class(technique_name)
            if technique_class is None:
                raise ValueError(
                    f"Technique '{technique_name}' not found. "
                    f"Available techniques: {self.list_available_techniques()}"
                )

            # Apply technique
            technique = technique_class()
            X_balanced, y_balanced = technique.balance(X_train, y_train)

            # Store balanced data for later export
            self.current_balanced_datasets[technique_name] = {
                "X_balanced": X_balanced,
                "y_balanced": y_balanced,
            }

            # Calculate metrics
            metrics = get_metrics(X_balanced, y_balanced, X_test, y_test)
            results[technique_name] = metrics

        self.results = results

        if plot_results:
            plot_comparison_results(results)

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
            include_plots: Whether to save visualization plots
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

            print(f"Saved balanced dataset for '{technique}' to {file_path}")

    def compare_balanced_class_distributions(
        self,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Compare class distributions of balanced datasets for all techniques.

        Args:
            save_path: Path to save the visualization (optional).

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

        # Call the visualization function
        plot_class_distributions_comparison(
            distributions,
            title="Class Distribution Comparison After Balancing",
            save_path=save_path,
        )

    def generate_learning_curves(
        self,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Generate and plot learning curves for multiple balancing techniques.

        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.current_balanced_datasets:
            raise ValueError(
                "No balanced datasets available. Run compare_techniques first."
            )

        learning_curve_data = get_learning_curve_data_multiple_techniques(
            techniques_data=self.current_balanced_datasets
        )

        plot_learning_curves(learning_curve_data, save_path=save_path)

    def _get_class_distribution(self) -> Dict[Any, int]:
        """Get the distribution of classes in the target variable."""
        return self.preprocessor.inspect_class_distribution(self.y)

    def _handle_quality_issues(self, quality_report: Dict[str, Any]) -> None:
        """Handle any data quality issues found."""
        warnings = []

        if quality_report["missing_values"].any():
            warnings.append(
                f"Data contains missing values: {quality_report['missing_values']}"
            )

        if quality_report["constant_features"].size > 0:
            warnings.append(
                f"Features {quality_report['constant_features']} have constant values"
            )

        if quality_report["feature_correlations"]:
            warnings.append(
                "Found highly correlated features: "
                f"{quality_report['feature_correlations']}"
            )

        if warnings:
            print("Data Quality Warnings:")
            for warning in warnings:
                print(f"- {warning}")
