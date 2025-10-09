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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Text preprocessing utilities"""
    
    def __init__(self, remove_stopwords=True, lowercase=True):
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.stemmer = PorterStemmer()
        
        # Download NLTK data if not already present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text)
        if self.lowercase:
            text = text.lower()
        
        # Remove special characters and digits, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if specified
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]
        
        # Stem words
        tokens = [self.stemmer.stem(word) for word in tokens]
        
        # Remove very short words (less than 2 characters)
        tokens = [word for word in tokens if len(word) > 1]
        
        return ' '.join(tokens)

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def load_and_clean_data(params):
    """Load and perform initial data cleaning"""
    logger.info("Loading raw data...")
    
    # Load the dataset
    df = pd.read_csv(params['data']['raw_data_path'], sep='\t')
    
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Remove samples with missing text or SDG labels
    initial_size = len(df)
    df = df.dropna(subset=['text', 'sdg'])
    logger.info(f"Removed {initial_size - len(df)} samples with missing text or SDG labels")
    
    # Filter out samples with very short text (less than 10 characters)
    df = df[df['text'].str.len() >= 10]
    logger.info(f"Final dataset size: {len(df)} samples")
    
    # Display SDG distribution
    sdg_counts = df['sdg'].value_counts().sort_index()
    logger.info(f"SDG distribution:\n{sdg_counts}")
    
    return df

def preprocess_features_and_labels(df, params):
    """Preprocess text features and encode labels"""
    logger.info("Preprocessing text features...")
    
    # Initialize text preprocessor
    preprocessor = TextPreprocessor(
        remove_stopwords=params['preprocessing']['remove_stopwords'],
        lowercase=params['preprocessing']['lowercase']
    )
    
    # Clean text data
    df['text_cleaned'] = df['text'].apply(preprocessor.clean_text)
    
    # Remove samples where cleaned text is empty
    df = df[df['text_cleaned'].str.len() > 0]
    logger.info(f"After text cleaning: {len(df)} samples")
    
    # Prepare features and labels
    X = df['text_cleaned'].values
    y = df['sdg'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    logger.info(f"Number of unique SDG classes: {len(label_encoder.classes_)}")
    logger.info(f"SDG classes: {label_encoder.classes_}")
    
    return X, y_encoded, label_encoder

def create_text_vectorizer(X_train, params):
    """Create and fit TF-IDF vectorizer"""
    logger.info("Creating TF-IDF vectorizer...")
    
    vectorizer = TfidfVectorizer(
        max_features=params['preprocessing']['max_features'],
        min_df=params['preprocessing']['min_df'],
        max_df=params['preprocessing']['max_df'],
        ngram_range=tuple(params['preprocessing']['ngram_range']),
        stop_words='english' if params['preprocessing']['remove_stopwords'] else None
    )
    
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    logger.info(f"Feature matrix shape: {X_train_vectorized.shape}")
    
    return vectorizer, X_train_vectorized

def split_data(X, y, params):
    """Split data into train, validation, and test sets"""
    logger.info("Splitting data...")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=params['data']['test_size'],
        random_state=params['data']['random_state'],
        stratify=y
    )
    
    # Second split: separate train and validation from remaining data
    val_size_adjusted = params['data']['val_size'] / (1 - params['data']['test_size'])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=params['data']['random_state'],
        stratify=y_temp
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def compute_and_save_statistics(X_train, X_val, X_test, y_train, y_val, y_test, 
                                label_encoder, params):
    """Compute and save dataset statistics"""
    logger.info("Computing dataset statistics...")
    
    # Compute class weights for imbalanced dataset
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    # Create statistics dictionary
    stats = {
        'dataset_info': {
            'total_samples': len(X_train) + len(X_val) + len(X_test),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'num_classes': len(label_encoder.classes_),
            'class_names': label_encoder.classes_.tolist()
        },
        'class_distribution': {
            'train': {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
            'val': {str(k): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))},
            'test': {str(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))}
        },
        'class_weights': {str(k): float(v) for k, v in class_weight_dict.items()},
        'preprocessing_params': params['preprocessing']
    }
    
    # Save statistics
    os.makedirs(params['data']['processed_data_path'], exist_ok=True)
    stats_path = os.path.join(params['data']['processed_data_path'], 'data_stats.json')
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Statistics saved to {stats_path}")
    return stats

def save_processed_data(X_train_vec, X_val_vec, X_test_vec, y_train, y_val, y_test,
                       vectorizer, label_encoder, params):
    """Save all processed data"""
    logger.info("Saving processed data...")
    
    processed_path = params['data']['processed_data_path']
    os.makedirs(processed_path, exist_ok=True)
    
    # Save feature matrices
    with open(os.path.join(processed_path, 'X_train.pkl'), 'wb') as f:
        pickle.dump(X_train_vec, f)
    
    with open(os.path.join(processed_path, 'X_val.pkl'), 'wb') as f:
        pickle.dump(X_val_vec, f)
    
    with open(os.path.join(processed_path, 'X_test.pkl'), 'wb') as f:
        pickle.dump(X_test_vec, f)
    
    # Save labels
    with open(os.path.join(processed_path, 'y_train.pkl'), 'wb') as f:
        pickle.dump(y_train, f)
    
    with open(os.path.join(processed_path, 'y_val.pkl'), 'wb') as f:
        pickle.dump(y_val, f)
    
    with open(os.path.join(processed_path, 'y_test.pkl'), 'wb') as f:
        pickle.dump(y_test, f)
    
    # Save vectorizer and label encoder
    with open(os.path.join(processed_path, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open(os.path.join(processed_path, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    logger.info("All processed data saved successfully!")

def main():
    """Main preprocessing pipeline"""
    logger.info("Starting data preprocessing pipeline...")
    
    # Load parameters
    params = load_params()
    
    # Load and clean data
    df = load_and_clean_data(params)
    
    # Preprocess features and labels
    X, y_encoded, label_encoder = preprocess_features_and_labels(df, params)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y_encoded, params)
    
    # Create and fit vectorizer
    vectorizer, X_train_vec = create_text_vectorizer(X_train, params)
    
    # Transform validation and test sets
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    
    # Compute and save statistics
    stats = compute_and_save_statistics(
        X_train, X_val, X_test, y_train, y_val, y_test,
        label_encoder, params
    )
    
    # Save all processed data
    save_processed_data(
        X_train_vec, X_val_vec, X_test_vec, y_train, y_val, y_test,
        vectorizer, label_encoder, params
    )
    
    logger.info("Data preprocessing completed successfully!")
    logger.info(f"Summary: {stats['dataset_info']}")

if __name__ == "__main__":
    main()