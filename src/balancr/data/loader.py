import pandas as pd
import numpy as np
import logging
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

        # Check for missing target values
        missing_target_mask = data[target_column].isna()
        missing_target_count = missing_target_mask.sum()

        if missing_target_count > 0:
            # Get the row numbers with missing target values (0-based index)
            missing_target_rows = data.index[missing_target_mask].tolist()

            # Log warning with list of rows that will be removed
            if len(missing_target_rows) <= 20:  # Show all rows if 20 or fewer
                logging.warning(
                    f"Rows with missing target values found and will be removed: {missing_target_rows}"
                )
            else:  # Show summary if more than 20 rows
                logging.warning(
                    f"Found {missing_target_count} rows with missing target values that will be removed. "
                    f"First few row indices: {missing_target_rows[:10]}..."
                )

            # Remove rows with missing target values
            data = data.dropna(subset=[target_column])

            # Provide summary of the cleaning action
            logging.info(
                f"Removed {missing_target_count} rows with missing target values. "
                f"Remaining rows: {len(data)}"
            )

            # Warn if a large proportion of the dataset was removed
            original_row_count = len(data) + missing_target_count
            removed_percentage = (missing_target_count / original_row_count) * 100
            if removed_percentage > 10:
                logging.warning(
                    f"A significant portion of the dataset ({removed_percentage:.1f}%) "
                    f"was removed due to missing target values."
                )

        # Now extract the cleaned target variable
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
