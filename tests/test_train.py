"""
Unit tests for model training module
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import tempfile
import os
import pickle
import json

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

class TestModelTraining:
    """Test model training functionality"""
    
    def setup_method(self):
        """Setup test data for each test"""
        # Create mock training data
        np.random.seed(42)
        
        # Create sample text data
        self.sample_texts = [
            "renewable energy solar power sustainability",
            "education quality learning development",
            "gender equality women empowerment",
            "clean water sanitation health",
            "climate change environmental protection"
        ] * 10  # Repeat to have enough samples
        
        # Create corresponding SDG labels
        self.sample_labels = [7, 4, 5, 6, 13] * 10
        
        # Create vectorizer and transform data
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.X_sample = self.vectorizer.fit_transform(self.sample_texts)
        
        # Create label encoder
        self.label_encoder = LabelEncoder()
        self.y_sample = self.label_encoder.fit_transform(self.sample_labels)
    
    def test_random_forest_model_creation(self):
        """Test Random Forest model creation and training"""
        model = RandomForestClassifier(
            n_estimators=10,  # Small for testing
            max_depth=3,
            random_state=42
        )
        
        # Train model
        model.fit(self.X_sample, self.y_sample)
        
        # Test predictions
        predictions = model.predict(self.X_sample)
        
        assert len(predictions) == len(self.y_sample)
        assert all(pred in self.label_encoder.classes_ for pred in predictions)
    
    def test_model_serialization(self):
        """Test model saving and loading"""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(self.X_sample, self.y_sample)
        
        # Test pickle serialization
        with tempfile.NamedTemporaryFile(delete=False) as f:
            pickle.dump(model, f)
            temp_file = f.name
        
        try:
            # Load model back
            with open(temp_file, 'rb') as f:
                loaded_model = pickle.load(f)
            
            # Test that loaded model works
            original_pred = model.predict(self.X_sample)
            loaded_pred = loaded_model.predict(self.X_sample)
            
            np.testing.assert_array_equal(original_pred, loaded_pred)
        finally:
            os.unlink(temp_file)
    
    def test_model_performance_metrics(self):
        """Test calculation of performance metrics"""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(self.X_sample, self.y_sample)
        
        predictions = model.predict(self.X_sample)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_sample, predictions)
        f1_macro = f1_score(self.y_sample, predictions, average='macro')
        precision_macro = precision_score(self.y_sample, predictions, average='macro')
        recall_macro = recall_score(self.y_sample, predictions, average='macro')
        
        # Verify metrics are reasonable
        assert 0 <= accuracy <= 1
        assert 0 <= f1_macro <= 1
        assert 0 <= precision_macro <= 1
        assert 0 <= recall_macro <= 1
    
    def test_feature_importance_extraction(self):
        """Test feature importance extraction"""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(self.X_sample, self.y_sample)
        
        # Get feature importance
        importance = model.feature_importances_
        
        assert len(importance) == self.X_sample.shape[1]
        assert all(imp >= 0 for imp in importance)
        assert abs(sum(importance) - 1.0) < 1e-6  # Should sum to 1
    
    def test_cross_validation_scoring(self):
        """Test cross-validation functionality"""
        from sklearn.model_selection import cross_val_score
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, self.X_sample, self.y_sample,
            cv=3,  # Small CV for testing
            scoring='accuracy'
        )
        
        assert len(cv_scores) == 3
        assert all(0 <= score <= 1 for score in cv_scores)
    
    def test_model_parameters_storage(self):
        """Test storing model parameters"""
        model = RandomForestClassifier(
            n_estimators=10,
            max_depth=5,
            min_samples_split=2,
            random_state=42
        )
        
        params = model.get_params()
        
        assert 'n_estimators' in params
        assert 'max_depth' in params
        assert 'random_state' in params
        assert params['n_estimators'] == 10
        assert params['max_depth'] == 5

class TestModelValidation:
    """Test model validation functionality"""
    
    def setup_method(self):
        """Setup validation test data"""
        np.random.seed(42)
        
        # Create larger dataset for train/val split
        texts = ["sample text about sdg " + str(i % 16 + 1) for i in range(100)]
        labels = [i % 16 + 1 for i in range(100)]
        
        vectorizer = TfidfVectorizer(max_features=50)
        X = vectorizer.fit_transform(texts)
        
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
        
        # Split into train/val
        split_idx = 80
        self.X_train = X[:split_idx]
        self.X_val = X[split_idx:]
        self.y_train = y[:split_idx]
        self.y_val = y[split_idx:]
    
    def test_validation_split_sizes(self):
        """Test that validation split creates correct sizes"""
        assert self.X_train.shape[0] == 80
        assert self.X_val.shape[0] == 20
        assert len(self.y_train) == 80
        assert len(self.y_val) == 20
    
    def test_training_validation_consistency(self):
        """Test training and validation consistency"""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Train on training set
        model.fit(self.X_train, self.y_train)
        
        # Evaluate on both sets
        train_score = model.score(self.X_train, self.y_train)
        val_score = model.score(self.X_val, self.y_val)
        
        # Training score should be reasonable
        assert 0 <= train_score <= 1
        assert 0 <= val_score <= 1
        
        # For this simple test, scores should be > 0
        assert train_score > 0
        assert val_score >= 0

class TestModelExport:
    """Test model export functionality"""
    
    def test_model_metadata_creation(self):
        """Test creation of model metadata"""
        metadata = {
            'model_info': {
                'name': 'test_model',
                'version': '1.0.0',
                'algorithm': 'random_forest'
            },
            'performance_metrics': {
                'accuracy': 0.85,
                'f1_macro': 0.82
            },
            'preprocessing_info': {
                'vectorizer_type': 'TfidfVectorizer',
                'max_features': 1000
            }
        }
        
        # Test JSON serialization
        json_str = json.dumps(metadata)
        loaded_metadata = json.loads(json_str)
        
        assert loaded_metadata['model_info']['name'] == 'test_model'
        assert loaded_metadata['performance_metrics']['accuracy'] == 0.85
    
    def test_inference_pipeline_creation(self):
        """Test creation of inference pipeline"""
        # Create simple pipeline components
        vectorizer = TfidfVectorizer(max_features=50)
        texts = ["test text"] * 10
        vectorizer.fit(texts)
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = vectorizer.transform(texts)
        y = [1] * 10
        model.fit(X, y)
        
        label_encoder = LabelEncoder()
        label_encoder.fit([1, 2, 3])
        
        # Create inference function
        def inference_pipeline(text):
            X_vec = vectorizer.transform([text])
            pred = model.predict(X_vec)[0]
            return {'predicted_sdg': int(pred)}
        
        # Test inference
        result = inference_pipeline("test input text")
        
        assert 'predicted_sdg' in result
        assert isinstance(result['predicted_sdg'], int)

# Integration tests
def test_full_training_pipeline_mock():
    """Test full training pipeline with mocked data"""
    
    # Mock data loading
    with patch('pickle.load') as mock_load:
        # Setup mock data
        vectorizer = TfidfVectorizer(max_features=50)
        texts = ["renewable energy"] * 20
        X_mock = vectorizer.fit_transform(texts)
        y_mock = np.array([7] * 20)
        
        # Configure mock to return our test data
        mock_load.side_effect = [X_mock, X_mock, y_mock, y_mock, vectorizer, LabelEncoder()]
        
        # Mock file existence
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', create=True):
                # Test that we can create and train a model
                model = RandomForestClassifier(n_estimators=5, random_state=42)
                model.fit(X_mock, y_mock)
                
                # Verify model was trained
                assert hasattr(model, 'feature_importances_')
                assert model.n_estimators == 5

if __name__ == "__main__":
    pytest.main([__file__])