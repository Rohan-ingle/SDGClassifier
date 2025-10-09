"""
Unit tests for data preprocessing module
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os
import yaml

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.preprocess import TextPreprocessor, load_params

class TestTextPreprocessor:
    """Test the TextPreprocessor class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.preprocessor = TextPreprocessor(remove_stopwords=True, lowercase=True)
    
    def test_clean_text_basic(self):
        """Test basic text cleaning"""
        text = "This is a TEST with numbers 123 and symbols!"
        cleaned = self.preprocessor.clean_text(text)
        
        # Should be lowercase and without numbers/symbols
        assert cleaned.islower()
        assert "123" not in cleaned
        assert "!" not in cleaned
        assert len(cleaned) > 0
    
    def test_clean_text_empty(self):
        """Test cleaning empty or None text"""
        assert self.preprocessor.clean_text("") == ""
        assert self.preprocessor.clean_text(None) == ""
        assert self.preprocessor.clean_text(pd.NA) == ""
    
    def test_clean_text_stopwords(self):
        """Test stopword removal"""
        text = "The quick brown fox jumps over the lazy dog"
        cleaned = self.preprocessor.clean_text(text)
        
        # Common stopwords should be removed
        assert "the" not in cleaned.lower()
        assert "over" not in cleaned.lower()
        
        # Content words should remain
        assert "fox" in cleaned or "jump" in cleaned  # May be stemmed
    
    def test_clean_text_without_stopwords(self):
        """Test preprocessing without stopword removal"""
        preprocessor = TextPreprocessor(remove_stopwords=False, lowercase=True)
        text = "The quick brown fox"
        cleaned = preprocessor.clean_text(text)
        
        # Stopwords should be preserved
        assert len(cleaned.split()) >= 3  # Should have most words

class TestLoadParams:
    """Test parameter loading"""
    
    def test_load_params_structure(self):
        """Test that params are loaded with correct structure"""
        # Create a temporary params file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            test_params = {
                'data': {'test_size': 0.2},
                'preprocessing': {'max_features': 1000},
                'model': {'algorithm': 'random_forest'}
            }
            yaml.dump(test_params, f)
            temp_file = f.name
        
        try:
            # Mock the file reading
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = yaml.dump(test_params)
                
                params = load_params()
                
                assert 'data' in params
                assert 'preprocessing' in params
                assert 'model' in params
                assert params['data']['test_size'] == 0.2
        finally:
            os.unlink(temp_file)

class TestDataValidation:
    """Test data validation functions"""
    
    def test_valid_dataframe_structure(self):
        """Test validation of dataframe structure"""
        # Create a mock dataframe with correct structure
        df = pd.DataFrame({
            'doi': ['10.1000/test1', '10.1000/test2'],
            'text_id': ['id1', 'id2'],
            'text': ['This is sample text about energy', 'Another text about education'],
            'sdg': [7, 4],
            'labels_negative': [1, 2],
            'labels_positive': [8, 7],
            'agreement': [0.8, 0.9]
        })
        
        # Check required columns exist
        required_cols = ['doi', 'text_id', 'text', 'sdg', 'labels_negative', 'labels_positive', 'agreement']
        assert all(col in df.columns for col in required_cols)
        
        # Check data types
        assert df['text'].dtype == 'object'
        assert df['sdg'].dtype in ['int64', 'int32']
    
    def test_invalid_dataframe_missing_columns(self):
        """Test handling of dataframe with missing columns"""
        df = pd.DataFrame({
            'text': ['Sample text'],
            'wrong_column': [1]
        })
        
        required_cols = ['doi', 'text_id', 'text', 'sdg', 'labels_negative', 'labels_positive', 'agreement']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        assert len(missing_cols) > 0
        assert 'sdg' in missing_cols

class TestFeatureEngineering:
    """Test feature engineering functions"""
    
    def test_text_length_distribution(self):
        """Test text length calculations"""
        texts = [
            "Short text",
            "This is a much longer text with many more words to test the length calculation",
            "Medium length text here"
        ]
        
        lengths = [len(text.split()) for text in texts]
        
        assert min(lengths) < max(lengths)
        assert all(length > 0 for length in lengths)
    
    def test_sdg_distribution(self):
        """Test SDG label distribution"""
        sdgs = [1, 2, 3, 1, 2, 1, 4, 5, 1]
        unique_sdgs, counts = np.unique(sdgs, return_counts=True)
        
        assert len(unique_sdgs) == 5  # Should have 5 unique SDGs
        assert max(counts) == 4  # SDG 1 appears 4 times
        assert all(sdg >= 1 and sdg <= 16 for sdg in unique_sdgs)

# Integration test
def test_preprocessing_pipeline_integration():
    """Test the complete preprocessing pipeline with mock data"""
    
    # Create mock data
    mock_data = pd.DataFrame({
        'doi': ['10.1000/test' + str(i) for i in range(10)],
        'text_id': [f'id{i}' for i in range(10)],
        'text': [
            'Renewable energy solutions for sustainable development',
            'Education quality improvement in developing countries',
            'Gender equality in workplace environments',
            'Clean water access for rural communities',
            'Climate change mitigation strategies',
            'Economic growth and employment opportunities',
            'Healthcare system strengthening initiatives',
            'Innovation and infrastructure development',
            'Reducing inequality in urban areas',
            'Peace and justice institutional building'
        ],
        'sdg': [7, 4, 5, 6, 13, 8, 3, 9, 10, 16],
        'labels_negative': [1] * 10,
        'labels_positive': [8] * 10,
        'agreement': [0.8] * 10
    })
    
    # Test preprocessor
    preprocessor = TextPreprocessor()
    
    # Clean all texts
    cleaned_texts = [preprocessor.clean_text(text) for text in mock_data['text']]
    
    # Verify cleaning worked
    assert all(len(text) > 0 for text in cleaned_texts)
    assert all(text.islower() for text in cleaned_texts if text)
    
    # Verify SDG distribution
    sdg_counts = mock_data['sdg'].value_counts()
    assert len(sdg_counts) == 10  # All unique SDGs
    assert all(sdg >= 1 and sdg <= 16 for sdg in mock_data['sdg'])

if __name__ == "__main__":
    pytest.main([__file__])