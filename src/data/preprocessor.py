from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
)
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """Handles data preprocessing operations"""

    def __init__(self):
        self.scaler = None
        self.imputer = None
        self.label_encoder = None
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None

    def inspect_class_distribution(self, y: np.ndarray) -> Dict[Any, int]:
        """
        Inspect the distribution of classes in the target variable

        Args:
            y: Target vector

        Returns:
            Dictionary mapping class labels to their counts
        """
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))

    def check_data_quality(
        self, X: np.ndarray, feature_names: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Check data quality issues

        Args:
            X: Feature matrix
            feature_names: Optional list of feature names

        Returns:
            Dictionary containing quality metrics
        """
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

        quality_report = {
            "missing_values": np.isnan(X).sum(axis=0),
            "constant_features": np.where(np.std(X, axis=0) == 0)[0],
            "feature_correlations": None,
        }

        # Calculate correlations if we have enough samples
        if X.shape[0] > 1:
            correlations = pd.DataFrame(X, columns=feature_names).corr()
            # Find highly correlated features (above 0.95)
            high_corr = np.where(np.abs(correlations) > 0.95)
            quality_report["feature_correlations"] = [
                (feature_names[i], feature_names[j], correlations.iloc[i, j])
                for i, j in zip(*high_corr)
                if i < j  # Only take upper triangle to avoid duplicates
            ]

        return quality_report

    def preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        handle_missing: str = "mean",
        scale: str = "standard",
        encode: str = "auto",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data with enhanced options

        Args:
            X: Feature matrix
            y: Target vector
            handle_missing: Strategy to handle missing values
                ("drop", "mean", "median", "mode", "none")
            scale: Scaling method
                ("standard", "minmax", "robust", "none")
            encode: Encoding method for categorical features
                ("auto", "onehot", "label", "ordinal", "none")

        Returns:
            Preprocessed X and y
        """
        # Convert to DataFrame for more flexible processing if not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Handle missing values
        if handle_missing != "none" and X.isna().any().any():
            if handle_missing == "drop":
                # Remove rows with any missing values
                mask = ~X.isna().any(axis=1)
                X = X[mask]
                y = y[mask] if isinstance(y, pd.Series) else y[mask]
            else:
                # Use SimpleImputer for other strategies
                strategy = (
                    handle_missing
                    if handle_missing in ["mean", "median", "most_frequent"]
                    else "mean"
                )
                if handle_missing == "mode":
                    strategy = "most_frequent"

                imputer = SimpleImputer(strategy=strategy)
                X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Apply scaling if requested
        if scale != "none":
            if scale == "standard":
                scaler = StandardScaler()
            elif scale == "minmax":
                scaler = MinMaxScaler()
            elif scale == "robust":
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()  # Default

            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Encode labels if necessary
        if encode != "none":
            # Determine if y needs encoding
            if not np.issubdtype(y.dtype, np.number):
                if encode in ["auto", "label"]:
                    # Use label encoding for the target
                    self.label_encoder = LabelEncoder()
                    y = self.label_encoder.fit_transform(y)

        # Return as numpy arrays to maintain compatibility
        return X.values, y
