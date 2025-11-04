"""
Data preprocessing pipeline for SDG Classification
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import yaml
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import logging

# Import modular feature engineering
from .feature_engineering import create_text_features, FeatureEngineer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Text preprocessing utilities"""

    def __init__(self, remove_stopwords=True, lowercase=True):
        """
        Initialize text preprocessor.

        Args:
            remove_stopwords: Whether to remove English stopwords
            lowercase: Whether to convert text to lowercase
        """
        logger.info("Initializing TextPreprocessor...")
        logger.info("  remove_stopwords: %s", remove_stopwords)
        logger.info("  lowercase: %s", lowercase)

        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.stemmer = PorterStemmer()

        # Download NLTK data if not already present
        try:
            nltk.data.find("tokenizers/punkt")
            logger.debug("NLTK punkt tokenizer already available")
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download("punkt")
            logger.info("NLTK punkt downloaded")

        try:
            nltk.data.find("corpora/stopwords")
            logger.debug("NLTK stopwords already available")
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download("stopwords")
            logger.info("NLTK stopwords downloaded")

        if self.remove_stopwords:
            self.stop_words = set(stopwords.words("english"))
            logger.info("Loaded %d English stopwords", len(self.stop_words))

    def clean_text(self, text):
        """
        Clean and preprocess text.

        Steps:
        1. Convert to lowercase (if enabled)
        2. Remove special characters and digits
        3. Tokenize
        4. Remove stopwords (if enabled)
        5. Apply stemming
        6. Remove short words

        Args:
            text: Raw text string

        Returns:
            Cleaned and preprocessed text string
        """
        if pd.isna(text):
            return ""

        original_length = len(str(text))

        # Convert to string and lowercase
        text = str(text)
        if self.lowercase:
            text = text.lower()

        # Remove special characters and digits, keep only letters and spaces
        text = re.sub(r"[^a-zA-Z\s]", " ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Tokenize
        tokens = word_tokenize(text)
        original_token_count = len(tokens)

        # Remove stopwords if specified
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]

        # Stem words
        tokens = [self.stemmer.stem(word) for word in tokens]

        # Remove very short words (less than 2 characters)
        tokens = [word for word in tokens if len(word) > 1]

        final_token_count = len(tokens)
        cleaned_text = " ".join(tokens)

        # Log detailed stats for first few texts (debug mode)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Text cleaning: %d chars -> %d chars, %d tokens -> %d tokens",
                original_length,
                len(cleaned_text),
                original_token_count,
                final_token_count,
            )

        return cleaned_text


def load_params():
    """
    Load parameters from params.yaml configuration file.

    Returns:
        Dictionary containing all configuration parameters
    """
    logger.info("Loading parameters from params.yaml...")

    try:
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)

        logger.info("Parameters loaded successfully")
        logger.debug("Available parameter sections: %s", list(params.keys()))

        return params
    except FileNotFoundError:
        logger.error("params.yaml file not found in current directory")
        raise
    except yaml.YAMLError as e:
        logger.error("Error parsing params.yaml: %s", str(e))
        raise
    except Exception as e:
        logger.error("Unexpected error loading parameters: %s", str(e))
        raise


def load_and_clean_data(params):
    """Load and perform initial data cleaning"""
    logger.info("Loading raw data...")

    # Load the dataset
    df = pd.read_csv(params["data"]["raw_data_path"], sep="\t")

    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Columns: {df.columns.tolist()}")

    # Remove samples with missing text or SDG labels
    initial_size = len(df)
    df = df.dropna(subset=["text", "sdg"])
    logger.info(f"Removed {initial_size - len(df)} samples with missing text or SDG labels")

    # Filter out samples with very short text (less than 10 characters)
    df = df[df["text"].str.len() >= 10]
    logger.info(f"Final dataset size: {len(df)} samples")

    # Display SDG distribution
    sdg_counts = df["sdg"].value_counts().sort_index()
    logger.info(f"SDG distribution:\n{sdg_counts}")

    return df


def preprocess_features_and_labels(df, params):
    """
    Preprocess text features and encode labels.

    Args:
        df: DataFrame containing text and SDG columns
        params: Dictionary containing preprocessing parameters

    Returns:
        Tuple of (X, y_encoded, label_encoder)
    """
    logger.info("Preprocessing text features and encoding labels...")
    initial_samples = len(df)

    # Initialize text preprocessor
    logger.info("Initializing text preprocessor...")
    preprocessor = TextPreprocessor(
        remove_stopwords=params["preprocessing"]["remove_stopwords"],
        lowercase=params["preprocessing"]["lowercase"],
    )

    # Clean text data
    logger.info("Applying text cleaning to %d samples...", len(df))
    df["text_cleaned"] = df["text"].apply(preprocessor.clean_text)
    logger.info("Text cleaning applied to all samples")

    # Remove samples where cleaned text is empty
    empty_text_count = (df["text_cleaned"].str.len() == 0).sum()
    if empty_text_count > 0:
        logger.warning("Found %d samples with empty cleaned text, removing...", empty_text_count)

    df = df[df["text_cleaned"].str.len() > 0]
    logger.info("After text cleaning: %d samples (removed %d)", len(df), initial_samples - len(df))

    # Prepare features and labels
    X = df["text_cleaned"].values
    y = df["sdg"].values

    logger.info("Feature and label arrays created")
    logger.info("  X shape: %s", X.shape)
    logger.info("  y shape: %s", y.shape)

    # Encode labels
    logger.info("Encoding SDG labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    logger.info("Label encoding complete")
    logger.info("  Number of unique SDG classes: %d", len(label_encoder.classes_))
    logger.info("  SDG classes: %s", label_encoder.classes_.tolist())
    logger.info("  Encoded label range: [%d, %d]", y_encoded.min(), y_encoded.max())

    return X, y_encoded, label_encoder


def create_text_vectorizer(X_train, params):
    """
    Create and fit TF-IDF vectorizer.

    This function is now a wrapper around the modular FeatureEngineer class.

    Args:
        X_train: Training text data
        params: Dictionary containing preprocessing parameters

    Returns:
        Tuple of (fitted vectorizer, transformed training data)
    """
    logger.info("Creating TF-IDF vectorizer using FeatureEngineer...")

    try:
        # Initialize feature engineer with preprocessing params
        feature_engineer = FeatureEngineer(params["preprocessing"])

        # Fit vectorizer on training data
        vectorizer, X_train_vectorized = feature_engineer.fit_vectorizer(X_train)

        logger.info("TF-IDF vectorizer created successfully")

        return vectorizer, X_train_vectorized
    except Exception as e:
        logger.error("Failed to create text vectorizer: %s", str(e))
        raise


def split_data(X, y, params):
    """Split data into train, validation, and test sets"""
    logger.info("Splitting data...")

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
        stratify=y,
    )

    # Second split: separate train and validation from remaining data
    val_size_adjusted = params["data"]["val_size"] / (1 - params["data"]["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_adjusted,
        random_state=params["data"]["random_state"],
        stratify=y_temp,
    )

    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_and_save_statistics(
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, params
):
    """Compute and save dataset statistics"""
    logger.info("Computing dataset statistics...")

    # Compute class weights for imbalanced dataset
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)

    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    # Create statistics dictionary
    stats = {
        "dataset_info": {
            "total_samples": len(X_train) + len(X_val) + len(X_test),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "num_classes": len(label_encoder.classes_),
            "class_names": label_encoder.classes_.tolist(),
        },
        "class_distribution": {
            "train": {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
            "val": {str(k): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))},
            "test": {str(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
        },
        "class_weights": {str(k): float(v) for k, v in class_weight_dict.items()},
        "preprocessing_params": params["preprocessing"],
    }

    # Save statistics
    os.makedirs(params["data"]["processed_data_path"], exist_ok=True)
    stats_path = os.path.join(params["data"]["processed_data_path"], "data_stats.json")

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Statistics saved to {stats_path}")
    return stats


def save_processed_data(
    X_train_vec, X_val_vec, X_test_vec, y_train, y_val, y_test, vectorizer, label_encoder, params
):
    """
    Save all processed data to disk.

    Saves:
    - Feature matrices (X_train, X_val, X_test)
    - Label arrays (y_train, y_val, y_test)
    - TF-IDF vectorizer
    - Label encoder

    Args:
        X_train_vec: Vectorized training features
        X_val_vec: Vectorized validation features
        X_test_vec: Vectorized test features
        y_train: Training labels
        y_val: Validation labels
        y_test: Test labels
        vectorizer: Fitted TF-IDF vectorizer
        label_encoder: Fitted label encoder
        params: Configuration parameters
    """
    logger.info("Saving processed data to disk...")

    processed_path = params["data"]["processed_data_path"]
    logger.info("Output directory: %s", processed_path)

    # Create directory if it doesn't exist
    os.makedirs(processed_path, exist_ok=True)
    logger.debug("Output directory created/verified")

    # Save feature matrices
    logger.info("Saving feature matrices...")
    files_saved = []

    train_path = os.path.join(processed_path, "X_train.pkl")
    with open(train_path, "wb") as f:
        pickle.dump(X_train_vec, f)
    logger.debug("Saved X_train.pkl (%s)", X_train_vec.shape)
    files_saved.append(("X_train.pkl", os.path.getsize(train_path)))

    val_path = os.path.join(processed_path, "X_val.pkl")
    with open(val_path, "wb") as f:
        pickle.dump(X_val_vec, f)
    logger.debug("Saved X_val.pkl (%s)", X_val_vec.shape)
    files_saved.append(("X_val.pkl", os.path.getsize(val_path)))

    test_path = os.path.join(processed_path, "X_test.pkl")
    with open(test_path, "wb") as f:
        pickle.dump(X_test_vec, f)
    logger.debug("Saved X_test.pkl (%s)", X_test_vec.shape)
    files_saved.append(("X_test.pkl", os.path.getsize(test_path)))

    # Save labels
    logger.info("Saving label arrays...")

    y_train_path = os.path.join(processed_path, "y_train.pkl")
    with open(y_train_path, "wb") as f:
        pickle.dump(y_train, f)
    logger.debug("Saved y_train.pkl (%d samples)", len(y_train))
    files_saved.append(("y_train.pkl", os.path.getsize(y_train_path)))

    y_val_path = os.path.join(processed_path, "y_val.pkl")
    with open(y_val_path, "wb") as f:
        pickle.dump(y_val, f)
    logger.debug("Saved y_val.pkl (%d samples)", len(y_val))
    files_saved.append(("y_val.pkl", os.path.getsize(y_val_path)))

    y_test_path = os.path.join(processed_path, "y_test.pkl")
    with open(y_test_path, "wb") as f:
        pickle.dump(y_test, f)
    logger.debug("Saved y_test.pkl (%d samples)", len(y_test))
    files_saved.append(("y_test.pkl", os.path.getsize(y_test_path)))

    # Save vectorizer and label encoder
    logger.info("Saving vectorizer and label encoder...")

    vec_path = os.path.join(processed_path, "vectorizer.pkl")
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.debug("Saved vectorizer.pkl")
    files_saved.append(("vectorizer.pkl", os.path.getsize(vec_path)))

    encoder_path = os.path.join(processed_path, "label_encoder.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    logger.debug("Saved label_encoder.pkl")
    files_saved.append(("label_encoder.pkl", os.path.getsize(encoder_path)))

    # Log summary of saved files
    logger.info("All processed data saved successfully!")
    logger.info("Saved %d files:", len(files_saved))
    total_size = 0
    for filename, size in files_saved:
        size_mb = size / (1024 * 1024)
        logger.info("  - %s: %.2f MB", filename, size_mb)
        total_size += size
    logger.info("Total size: %.2f MB", total_size / (1024 * 1024))


def main():
    """
    Main preprocessing pipeline.

    Executes the complete preprocessing workflow:
    1. Load parameters and raw data
    2. Clean and preprocess text
    3. Encode labels
    4. Split into train/val/test sets
    5. Create TF-IDF features using modular feature engineering
    6. Compute statistics
    7. Save all processed data
    """
    logger.info("=" * 70)
    logger.info("Starting Data Preprocessing Pipeline")
    logger.info("=" * 70)

    try:
        # Load parameters
        logger.info("Step 1/7: Loading parameters...")
        params = load_params()
        logger.info("Parameters loaded successfully")
        logger.debug("Preprocessing params: %s", params.get("preprocessing", {}))

        # Load and clean data
        logger.info("Step 2/7: Loading and cleaning raw data...")
        df = load_and_clean_data(params)
        logger.info("Data loaded and cleaned: %d samples", len(df))

        # Preprocess features and labels
        logger.info("Step 3/7: Preprocessing features and encoding labels...")
        X, y_encoded, label_encoder = preprocess_features_and_labels(df, params)
        logger.info("Features preprocessed, labels encoded")

        # Split data
        logger.info("Step 4/7: Splitting data into train/val/test sets...")
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y_encoded, params)
        logger.info("Data split complete")

        # Create features using modular feature engineering
        logger.info("Step 5/7: Creating TF-IDF features using modular feature engineering...")
        X_train_vec, X_val_vec, X_test_vec, feature_engineer = create_text_features(
            X_train, X_val, X_test, params["preprocessing"]
        )
        logger.info("Feature engineering complete")

        # Get the vectorizer from the feature engineer
        vectorizer = feature_engineer.vectorizer

        # Compute and save statistics
        logger.info("Step 6/7: Computing and saving statistics...")
        stats = compute_and_save_statistics(
            X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, params
        )
        logger.info("Statistics computed and saved")

        # Save all processed data
        logger.info("Step 7/7: Saving all processed data...")
        save_processed_data(
            X_train_vec,
            X_val_vec,
            X_test_vec,
            y_train,
            y_val,
            y_test,
            vectorizer,
            label_encoder,
            params,
        )
        logger.info("All processed data saved")

        # Log final summary
        logger.info("=" * 70)
        logger.info("Data Preprocessing Pipeline Completed Successfully!")
        logger.info("=" * 70)
        logger.info("Summary Statistics:")
        logger.info("  Total samples: %d", stats["dataset_info"]["total_samples"])
        logger.info("  Train samples: %d", stats["dataset_info"]["train_samples"])
        logger.info("  Validation samples: %d", stats["dataset_info"]["val_samples"])
        logger.info("  Test samples: %d", stats["dataset_info"]["test_samples"])
        logger.info("  Number of classes: %d", stats["dataset_info"]["num_classes"])
        logger.info("  Feature dimensions: %d", X_train_vec.shape[1])
        logger.info("=" * 70)

    except Exception as e:
        logger.error("=" * 70)
        logger.error("Preprocessing pipeline failed!")
        logger.error("Error: %s", str(e))
        logger.exception("Full traceback:")
        logger.error("=" * 70)
        raise


if __name__ == "__main__":
    main()
