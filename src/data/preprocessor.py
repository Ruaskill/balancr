from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataPreprocessor:
    """Handles data preprocessing operations"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None

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
        scale_features: bool = True,
        encode_labels: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data

        Args:
            X: Feature matrix
            y: Target vector
            scale_features: Whether to scale features
            encode_labels: Whether to encode categorical labels

        Returns:
            Preprocessed X and y
        """
        # Handle missing values
        if np.isnan(X).any():
            # Replace missing values with mean of column
            X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))

        # Scale features
        if scale_features:
            X = self.scaler.fit_transform(X)

        # Encode labels if they're not already numeric
        if encode_labels and not np.issubdtype(y.dtype, np.number):
            y = self.label_encoder.fit_transform(y)

        return X, y
