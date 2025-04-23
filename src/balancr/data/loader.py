import pandas as pd
import numpy as np
from typing import Tuple, Union, Optional
from pathlib import Path


class DataLoader:
    """Handles loading data from various file formats"""

    @staticmethod
    def load_data(
        file_path: Union[str, Path],
        target_column: str,
        feature_columns: Optional[list] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from various file formats (CSV, Excel)

        Args:
            file_path: Path to the data file
            target_column: Name of the target column
            feature_columns: List of feature columns to use (optional)

        Returns:
            X: Feature matrix
            y: Target vector
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() == ".csv":
            data = pd.read_csv(file_path)
        elif file_path.suffix.lower() in [".xlsx", ".xls"]:
            try:
                data = pd.read_excel(file_path)
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "The openpyxl package is required to read Excel files. "
                    "Please install it using: pip install openpyxl"
                )
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # Extract target variable
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        y = data[target_column].values

        # Extract features
        if feature_columns is None:
            # Use all columns except target
            feature_columns = [col for col in data.columns if col != target_column]
        else:
            # Verify all specified feature columns exist
            missing_cols = [col for col in feature_columns if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Feature columns not found: {missing_cols}")

        X = data[feature_columns].values
        return X, y
