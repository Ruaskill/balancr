import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import FeatureHasher


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
        self,
        X: np.ndarray,
        feature_names: Optional[list] = None,
        correlation_threshold: float = 0.95,
    ) -> Dict[str, list]:
        """
        Check data quality issues

        Args:
            X: Feature matrix (numpy array or pandas DataFrame)
            feature_names: Optional list of feature names

        Returns:
            Dictionary containing quality metrics with more descriptive information
        """
        # Validate correlation threshold
        if not (0 <= correlation_threshold <= 1):
            raise ValueError("correlation_threshold must be between 0 and 1")

        # Handle input that is already a DataFrame or convert numpy array to DataFrame
        if not isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X
            if feature_names is None:
                feature_names = X_df.columns.tolist()

        # Check for missing values and create a more descriptive result
        missing_values_counts = X_df.isna().sum()
        missing_values = []
        for feature_name, count in missing_values_counts.items():
            if count > 0:
                missing_values.append((feature_name, int(count)))

        quality_report = {
            "missing_values": missing_values,
            "constant_features": [],
            "feature_correlations": [],
        }

        # Check for constant features
        constant_features = []
        for i, col in enumerate(X_df.columns):
            if X_df[col].nunique(dropna=True) <= 1:
                constant_features.append((col, i))

        quality_report["constant_features"] = constant_features

        # Calculate correlations only on numeric columns
        numeric_cols = X_df.select_dtypes(include=np.number).columns

        if len(numeric_cols) > 1 and X_df.shape[0] > 1:
            try:
                correlations = X_df[numeric_cols].corr()
                high_corr_pairs = []

                for i in range(len(correlations.columns)):
                    for j in range(i + 1, len(correlations.columns)):
                        if abs(correlations.iloc[i, j]) > correlation_threshold:
                            # Map back to original column names
                            col_i = correlations.columns[i]
                            col_j = correlations.columns[j]
                            high_corr_pairs.append(
                                (col_i, col_j, correlations.iloc[i, j])
                            )
                quality_report["feature_correlations"] = high_corr_pairs
            except Exception:
                # In case of correlation calculation errors
                quality_report["feature_correlations"] = []
        else:
            quality_report["feature_correlations"] = []

        return quality_report

    def preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        handle_missing: str = "mean",
        scale: str = "standard",
        handle_constant_features: str = "none",
        handle_correlations: str = "none",
        constant_features: Optional[List[str]] = None,
        correlated_features: Optional[List[Tuple[str, str, float]]] = None,
        categorical_features: Dict[str, str] = None,
        all_features: List[str] = None,
        encode_target: bool = True,
        hash_components_dict: Dict[str, int] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Preprocess the data with enhanced options for categorical features.

        Args:
            X: Feature matrix
            y: Target vector
            handle_missing: Strategy to handle missing values
                ("drop", "mean", "median", "mode", "none")
            scale: Scaling method
                ("standard", "minmax", "robust", "none")
            categorical_features: Dictionary mapping categorical column names to encoding methods
                Each column will be encoded according to its specified method:
                "onehot", "label", "ordinal", or "none"
            all_features: List of all feature column names. If provided, these will be used
                as column names for the DataFrame.

        Returns:
            Preprocessed X (as DataFrame with proper column names) and y
        """
        # Initialise categorical_features dictionary if None
        if categorical_features is None:
            categorical_features = {}
        if hash_components_dict is None:
            hash_components_dict = {}

        # Convert to DataFrame for more flexible processing if not already
        if not isinstance(X, pd.DataFrame):
            if all_features and len(all_features) == X.shape[1]:
                # If all feature names are provided, use them as column names
                X = pd.DataFrame(X, columns=all_features)
            else:
                # Use generic column names if not provided
                X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        # Convert y to Series if it's not already
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # Store the initial feature names for reference
        self.feature_names = X.columns.tolist()

        # Handle constant features
        if (
            handle_constant_features == "drop"
            and constant_features
            and len(constant_features) > 0
        ):
            # Log the constant features being dropped
            logging.debug(
                f"Dropping {len(constant_features)} constant features: {constant_features}"
            )

            # Drop the constant features
            X = X.drop(columns=constant_features)

            # Update feature names
            self.feature_names = X.columns.tolist()

        # Handle highly correlated features
        if (
            handle_correlations != "none"
            and correlated_features
            and len(correlated_features) > 0
        ):
            # Create a list to track features to drop
            features_to_drop = []

            # Mapping for PCA components if needed
            pca_mapping = {}

            # Process each correlated pair
            for feature1, feature2, corr_value in correlated_features:
                # Skip if either feature has already been dropped
                if feature1 in features_to_drop or feature2 in features_to_drop:
                    continue

                if handle_correlations == "drop_first":
                    # Drop the first feature in the pair
                    features_to_drop.append(feature1)
                    logging.debug(
                        f"Dropping {feature1} (first feature in correlation pair"
                        "with {feature2}, corr={corr_value:.2f})"
                    )

                elif handle_correlations == "drop_lowest":
                    # Calculate variance of both features
                    var1 = X[feature1].var()
                    var2 = X[feature2].var()

                    # Drop the feature with lower variance
                    if var1 < var2:
                        features_to_drop.append(feature1)
                        logging.debug(
                            f"Dropping {feature1} (lower variance in correlation"
                            "pair with {feature2}, corr={corr_value:.2f})"
                        )
                    else:
                        features_to_drop.append(feature2)
                        logging.debug(
                            f"Dropping {feature2} (lower variance in correlation"
                            "pair with {feature1}, corr={corr_value:.2f})"
                        )

                elif handle_correlations == "pca":
                    # Only process each pair once
                    pair_key = tuple(sorted([feature1, feature2]))
                    if pair_key in pca_mapping:
                        continue

                    # Get the data for these two features
                    pair_data = X[[feature1, feature2]]

                    try:
                        # Apply PCA to reduce to one component
                        from sklearn.decomposition import PCA

                        pca = PCA(n_components=1)
                        pca_result = pca.fit_transform(pair_data)

                        # Create a new feature name for the PCA component
                        pca_feature = f"pca_{feature1}_{feature2}"

                        # Add the PCA component as a new column
                        X[pca_feature] = pca_result

                        # Mark both original features for dropping
                        features_to_drop.extend([feature1, feature2])

                        # Store the mapping for reference
                        pca_mapping[pair_key] = pca_feature

                        logging.debug(
                            f"Applied PCA to correlated features {feature1} and {feature2} (corr={corr_value:.2f})"
                        )
                    except Exception as e:
                        logging.warning(
                            f"Failed to apply PCA to {feature1} and {feature2}: {str(e)}"
                        )

            # Drop all marked features at once
            if features_to_drop:
                X = X.drop(columns=features_to_drop)
                logging.debug(
                    f"Dropped {len(features_to_drop)} features due to high correlation"
                )

            # Update feature names
            self.feature_names = X.columns.tolist()

        # Handle missing values
        if handle_missing != "none" and X.isna().any().any():
            if handle_missing == "drop":
                # Remove rows with any missing values
                mask = ~X.isna().any(axis=1)
                X = X[mask].copy()
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

                # Identify numerical columns (those not in categorical_features)
                numerical_cols = [
                    col for col in X.columns if col not in categorical_features
                ]

                # Apply imputation to numerical columns
                if numerical_cols:
                    imputer = SimpleImputer(strategy=strategy)
                    X[numerical_cols] = pd.DataFrame(
                        imputer.fit_transform(X[numerical_cols]),
                        columns=numerical_cols,
                        index=X.index,
                    )

                # For categorical columns with missing values, fill with mode
                for col in categorical_features:
                    if col in X.columns and X[col].isna().any():
                        X[col] = X[col].fillna(X[col].mode().iloc[0])

        # Apply scaling only to numerical features (those not in categorical_features)
        if scale != "none":
            numerical_cols = [
                col for col in X.columns if col not in categorical_features
            ]
            if numerical_cols:
                if scale == "standard":
                    scaler = StandardScaler()
                elif scale == "minmax":
                    scaler = MinMaxScaler()
                elif scale == "robust":
                    scaler = RobustScaler()
                else:
                    scaler = StandardScaler()  # Default

                X[numerical_cols] = pd.DataFrame(
                    scaler.fit_transform(X[numerical_cols]),
                    columns=numerical_cols,
                    index=X.index,
                )

        # Process categorical columns based on their specified encoding types
        for col, encoding_info in categorical_features.items():
            # Skip if column doesn't exist in the DataFrame
            if col not in X.columns:
                continue

            # Normalise encoding_info into (encoding_type, extra_info)
            if isinstance(encoding_info, list):
                encoding_type = encoding_info[0]
                extra_info = encoding_info[1:]
            else:
                encoding_type = encoding_info
                extra_info = []

            # Apply encoding based on type
            if encoding_type == "onehot":
                unique_count = X[col].nunique()
                if unique_count > 50:  # Threshold for "high cardinality"
                    logging.warning(
                        f"Column '{col}' has {unique_count} unique values. "
                        "One-hot encoding may create too many features."
                    )
                    if unique_count > 100:  # Very high cardinality threshold
                        logging.warning(
                            f"Falling back to hash encoding for column '{col}' "
                            "due to very high cardinality."
                        )
                        # Use 32 hash columns as a default fallback
                        n_components = 32

                        # Use feature hashing for high cardinality
                        hasher = FeatureHasher(
                            n_features=n_components, input_type="string"
                        )

                        # Convert column values to a format FeatureHasher can handle
                        feature_values = X[col].astype(str).tolist()
                        hashed_features = hasher.transform(
                            [[value] for value in feature_values]
                        )

                        # Create a DataFrame with descriptive column names
                        hashed_df = pd.DataFrame(
                            hashed_features.toarray(),
                            index=X.index,
                            columns=[f"{col}_hash_{i}" for i in range(n_components)],
                        )
                        # Drop original column and add hashed features
                        X = pd.concat([X.drop(col, axis=1), hashed_df], axis=1)
                    else:
                        # Proceed with one-hot encoding but warn the user
                        # Use descriptive column names: original_col_value
                        dummies = pd.get_dummies(
                            X[col], prefix=col, prefix_sep="_", drop_first=False
                        )
                        X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
                else:
                    # Normal one-hot encoding for manageable cardinality
                    # Use descriptive column names: original_col_value
                    dummies = pd.get_dummies(
                        X[col], prefix=col, prefix_sep="_", drop_first=False
                    )
                    X = pd.concat([X.drop(col, axis=1), dummies], axis=1)

            elif encoding_type == "hash":
                # Get the number of components for this feature, if specified
                n_components = (
                    extra_info[0] if extra_info else 32
                )  # Default to 32 if not specified

                hasher = FeatureHasher(n_features=n_components, input_type="string")

                # Convert column values to a format FeatureHasher can handle
                feature_values = X[col].astype(str).tolist()
                hashed_features = hasher.transform(
                    [[value] for value in feature_values]
                )

                # Create a DataFrame with descriptive column names
                hashed_df = pd.DataFrame(
                    hashed_features.toarray(),
                    index=X.index,
                    columns=[f"{col}_hash_{i}" for i in range(n_components)],
                )
                # Drop original column and add hashed features
                X = pd.concat([X.drop(col, axis=1), hashed_df], axis=1)

            elif encoding_type == "label":
                # Label encode this column
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])

            elif encoding_type == "ordinal":
                # Ordinal encode this column
                categories = X[col].unique()
                mapping = {cat: i for i, cat in enumerate(categories)}
                X[col] = X[col].map(mapping)

            # Skip if encoding type is "none"

        # Encode labels if necessary (for the target variable)
        if encode_target and not np.issubdtype(y.dtype, np.number):
            # Use label encoding for the target
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)

        # Store the column names for future reference
        self.feature_names = list(X.columns)

        # Return the DataFrame with column names preserved and the numpy array for y
        return X, y

    def assign_encoding_types(
        self,
        df,
        categorical_columns,
        encoding_type="auto",
        hash_components=32,
        ordinal_columns=None,
    ):
        """
        Assigns encoding types to categorical features based on user preference or automatic recommendation.

        Args:
            df: DataFrame containing the data
            categorical_columns: List of categorical feature column names
            encoding_type: Global encoding strategy: "auto", "onehot", "label", "ordinal", or "none"
            ordinal_columns: List of categorical columns that have a natural order

        Returns:
            Dictionary mapping column names to their assigned encoding types

        Raises:
            ValueError: If an invalid encoding type is provided
        """
        if not categorical_columns:
            return {}

        # Initialise encoding types dictionary
        encoding_types = {}

        # Initialise ordinal columns if None
        if ordinal_columns is None:
            ordinal_columns = []

        # Validate and filter categorical columns
        valid_categorical_columns = []
        for col in categorical_columns:
            if col in df.columns:
                valid_categorical_columns.append(col)
            else:
                logging.warning(
                    f"Column '{col}' not found in the dataset and will be ignored"
                )

        # If no valid columns remain, return empty dictionary
        if not valid_categorical_columns:
            logging.warning("No valid categorical columns found in the dataset")
            return {}

        # Assign specific encoding type to all columns
        if encoding_type in ["onehot", "label", "ordinal"]:
            for col in valid_categorical_columns:
                encoding_types[col] = encoding_type

        # Assign hash encoding with components to all columns
        elif encoding_type == "hash":
            for col in valid_categorical_columns:
                encoding_types[col] = ["hash", hash_components]

        # Assign "none" to all columns
        elif encoding_type == "none":
            for col in valid_categorical_columns:
                encoding_types[col] = "none"

        # Recommend encoding types based on data characteristics
        elif encoding_type == "auto":
            for col in valid_categorical_columns:
                # If column is marked as ordinal, assign ordinal encoding
                if col in ordinal_columns:
                    encoding_types[col] = "ordinal"
                    continue

                # Get cardinality (number of unique values)
                unique_count = df[col].nunique()

                # Check for highly skewed distribution
                value_counts = df[col].value_counts(normalize=True)
                is_highly_skewed = (
                    value_counts.iloc[0] >= 0.8
                )  # If dominant category > 80%

                # Check for many rare categories
                rare_categories = (value_counts < 0.05).sum()
                has_many_rare_cats = (
                    rare_categories > unique_count * 0.5
                )  # If >50% of categories are rare

                # Recommend based on cardinality and distribution characteristics
                if unique_count <= 20:
                    # For low cardinality, use one-hot
                    encoding_types[col] = "onehot"
                elif is_highly_skewed:
                    # For skewed distributions, use label encoding
                    encoding_types[col] = "label"
                elif has_many_rare_cats and unique_count <= 29:
                    # For many rare categories and under cardinality of 50, use label encoding
                    encoding_types[col] = "label"
                else:
                    # For high cardinality with balanced distribution, use hash encoding
                    # Adjust number of hash columns based on cardinality
                    n_components = min(32, max(16, int(unique_count / 4)))
                    encoding_types[col] = ["hash", n_components]

        else:
            raise ValueError(
                f"Invalid encoding type: {encoding_type}. Must be one of: auto, onehot, label, ordinal, hash, none"
            )

        return encoding_types
