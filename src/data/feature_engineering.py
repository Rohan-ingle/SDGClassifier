"""
Feature engineering module for SDG Classification

This module handles all feature engineering operations including:
- TF-IDF vectorization
- Text feature extraction
- Feature transformation and scaling
"""

import numpy as np
import pickle
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, Dict, Any

# Setup logging
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering class for text-based SDG classification.
    Handles TF-IDF vectorization and feature transformation.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the feature engineer with preprocessing parameters.

        Args:
            params: Dictionary containing preprocessing parameters
        """
        self.params = params
        self.vectorizer = None
        logger.info("FeatureEngineer initialized with parameters: %s", params)

    def create_tfidf_vectorizer(self) -> TfidfVectorizer:
        """
        Create a TF-IDF vectorizer with specified parameters.

        Returns:
            TfidfVectorizer: Configured TF-IDF vectorizer instance
        """
        logger.info("Creating TF-IDF vectorizer with parameters:")
        logger.info("  - max_features: %s", self.params.get("max_features"))
        logger.info("  - min_df: %s", self.params.get("min_df"))
        logger.info("  - max_df: %s", self.params.get("max_df"))
        logger.info("  - ngram_range: %s", self.params.get("ngram_range"))

        try:
            vectorizer = TfidfVectorizer(
                max_features=self.params.get("max_features"),
                min_df=self.params.get("min_df", 2),
                max_df=self.params.get("max_df", 0.95),
                ngram_range=tuple(self.params.get("ngram_range", [1, 2])),
                stop_words="english" if self.params.get("remove_stopwords", True) else None,
                sublinear_tf=True,  # Apply sublinear tf scaling
                use_idf=True,
                smooth_idf=True,
                norm="l2",
            )
            logger.info("TF-IDF vectorizer created successfully")
            return vectorizer
        except Exception as e:
            logger.error("Failed to create TF-IDF vectorizer: %s", str(e))
            raise

    def fit_vectorizer(self, X_train: np.ndarray) -> Tuple[TfidfVectorizer, Any]:
        """
        Fit the TF-IDF vectorizer on training data.

        Args:
            X_train: Training text data (array of strings)

        Returns:
            Tuple of (fitted vectorizer, transformed training data)
        """
        logger.info("Fitting TF-IDF vectorizer on training data...")
        logger.info("Training data shape: %s samples", len(X_train))

        try:
            self.vectorizer = self.create_tfidf_vectorizer()
            X_train_vectorized = self.vectorizer.fit_transform(X_train)

            logger.info("Vectorizer fitted successfully")
            logger.info("Vocabulary size: %d", len(self.vectorizer.vocabulary_))
            logger.info("Feature matrix shape: %s", X_train_vectorized.shape)
            logger.info(
                "Feature matrix density: %.2f%%",
                (
                    X_train_vectorized.nnz
                    / (X_train_vectorized.shape[0] * X_train_vectorized.shape[1])
                )
                * 100,
            )

            return self.vectorizer, X_train_vectorized
        except Exception as e:
            logger.error("Failed to fit vectorizer: %s", str(e))
            raise

    def transform_features(self, X: np.ndarray, dataset_name: str = "data") -> Any:
        """
        Transform text data using the fitted vectorizer.

        Args:
            X: Text data to transform (array of strings)
            dataset_name: Name of the dataset for logging purposes

        Returns:
            Transformed feature matrix
        """
        if self.vectorizer is None:
            logger.error("Vectorizer not fitted. Call fit_vectorizer first.")
            raise ValueError("Vectorizer must be fitted before transforming data")

        logger.info("Transforming %s features...", dataset_name)
        logger.info("Input data shape: %s samples", len(X))

        try:
            X_transformed = self.vectorizer.transform(X)
            logger.info("Transformation complete")
            logger.info("Output feature matrix shape: %s", X_transformed.shape)
            logger.info(
                "Output matrix density: %.2f%%",
                (X_transformed.nnz / (X_transformed.shape[0] * X_transformed.shape[1])) * 100,
            )

            return X_transformed
        except Exception as e:
            logger.error("Failed to transform features: %s", str(e))
            raise

    def get_feature_names(self) -> list:
        """
        Get the feature names from the fitted vectorizer.

        Returns:
            List of feature names
        """
        if self.vectorizer is None:
            logger.error("Vectorizer not fitted. No feature names available.")
            raise ValueError("Vectorizer must be fitted before getting feature names")

        try:
            if hasattr(self.vectorizer, "get_feature_names_out"):
                feature_names = self.vectorizer.get_feature_names_out()
            else:
                feature_names = self.vectorizer.get_feature_names()

            logger.info("Retrieved %d feature names", len(feature_names))
            return list(feature_names)
        except Exception as e:
            logger.error("Failed to get feature names: %s", str(e))
            raise

    def get_top_features_per_class(self, X: Any, y: np.ndarray, top_n: int = 10) -> Dict[int, list]:
        """
        Get top N features for each class based on TF-IDF scores.

        Args:
            X: Transformed feature matrix
            y: Label array
            top_n: Number of top features to return per class

        Returns:
            Dictionary mapping class labels to their top features
        """
        logger.info("Computing top %d features per class...", top_n)

        try:
            feature_names = self.get_feature_names()
            unique_classes = np.unique(y)
            top_features_per_class = {}

            for cls in unique_classes:
                # Get mean TF-IDF scores for this class
                class_mask = y == cls
                class_features = X[class_mask].mean(axis=0).A1  # Convert to 1D array

                # Get indices of top features
                top_indices = class_features.argsort()[-top_n:][::-1]
                top_feature_names = [feature_names[i] for i in top_indices]
                top_feature_scores = [class_features[i] for i in top_indices]

                top_features_per_class[int(cls)] = [
                    {"feature": name, "score": float(score)}
                    for name, score in zip(top_feature_names, top_feature_scores)
                ]

                logger.debug("Class %d top features: %s", cls, top_feature_names[:5])

            logger.info("Top features computed for %d classes", len(unique_classes))
            return top_features_per_class
        except Exception as e:
            logger.error("Failed to compute top features: %s", str(e))
            raise

    def save_vectorizer(self, filepath: str) -> None:
        """
        Save the fitted vectorizer to disk.

        Args:
            filepath: Path where to save the vectorizer
        """
        if self.vectorizer is None:
            logger.error("Cannot save vectorizer: not fitted yet")
            raise ValueError("Vectorizer must be fitted before saving")

        logger.info("Saving vectorizer to %s", filepath)

        try:
            with open(filepath, "wb") as f:
                pickle.dump(self.vectorizer, f)
            logger.info("Vectorizer saved successfully")
        except Exception as e:
            logger.error("Failed to save vectorizer: %s", str(e))
            raise

    def load_vectorizer(self, filepath: str) -> None:
        """
        Load a fitted vectorizer from disk.

        Args:
            filepath: Path to the saved vectorizer
        """
        logger.info("Loading vectorizer from %s", filepath)

        try:
            with open(filepath, "rb") as f:
                self.vectorizer = pickle.load(f)
            logger.info("Vectorizer loaded successfully")
            logger.info("Vocabulary size: %d", len(self.vectorizer.vocabulary_))
        except Exception as e:
            logger.error("Failed to load vectorizer: %s", str(e))
            raise


def create_text_features(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, params: Dict[str, Any]
) -> Tuple[Any, Any, Any, FeatureEngineer]:
    """
    High-level function to create text features for train, validation, and test sets.

    Args:
        X_train: Training text data
        X_val: Validation text data
        X_test: Test text data
        params: Preprocessing parameters

    Returns:
        Tuple of (X_train_transformed, X_val_transformed, X_test_transformed, feature_engineer)
    """
    logger.info("Starting feature engineering pipeline...")
    logger.info(
        "Train samples: %d, Val samples: %d, Test samples: %d",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    try:
        # Initialize feature engineer
        feature_engineer = FeatureEngineer(params)

        # Fit on training data
        vectorizer, X_train_vec = feature_engineer.fit_vectorizer(X_train)

        # Transform validation and test data
        X_val_vec = feature_engineer.transform_features(X_val, "validation")
        X_test_vec = feature_engineer.transform_features(X_test, "test")

        logger.info("Feature engineering pipeline completed successfully")

        return X_train_vec, X_val_vec, X_test_vec, feature_engineer
    except Exception as e:
        logger.error("Feature engineering pipeline failed: %s", str(e))
        raise


if __name__ == "__main__":
    # Configure logging for standalone testing
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.info("Feature engineering module loaded")
    logger.info("This module is designed to be imported, not run directly")
