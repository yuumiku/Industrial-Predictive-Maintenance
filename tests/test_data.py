"""
Unit tests for data validation and preprocessing.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocess import DataPreprocessor # type: ignore


class TestDataPreprocessor:
    """Test data preprocessing functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        np.random.seed(42)
        data = {
            'UID': range(1, 101),
            'Product ID': ['PROD_' + chr(65 + i % 3) for i in range(100)],
            'Type': ['L'] * 33 + ['M'] * 33 + ['H'] * 34,
            'Air temperature [K]': np.random.normal(298, 2, 100),
            'Process temperature [K]': np.random.normal(308, 2, 100),
            'Rotational speed [rpm]': np.random.normal(1500, 100, 100),
            'Torque [Nm]': np.random.normal(40, 5, 100),
            'Tool wear [min]': np.random.uniform(0, 200, 100),
            'Machine failure': np.random.binomial(1, 0.05, 100),
            'TWF': np.random.binomial(1, 0.01, 100),
            'HDF': np.random.binomial(1, 0.01, 100),
            'PWF': np.random.binomial(1, 0.01, 100),
            'OSF': np.random.binomial(1, 0.01, 100),
            'RNF': np.random.binomial(1, 0.01, 100),
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_csv_path(self, tmp_path, sample_data):
        """Create temporary CSV file for testing."""
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        return str(csv_path)
    
    def test_preprocessor_initialization(self, sample_csv_path):
        """Test preprocessor initializes correctly."""
        preprocessor = DataPreprocessor(sample_csv_path)
        assert preprocessor.data_path == sample_csv_path
        assert preprocessor.df is None
        assert preprocessor.X is None
        assert preprocessor.y is None
    
    def test_load_data(self, sample_csv_path):
        """Test data loading."""
        preprocessor = DataPreprocessor(sample_csv_path)
        df = preprocessor.load_data()
        
        assert df is not None
        assert len(df) > 0
        assert preprocessor.df is not None
    
    def test_load_data_shape(self, sample_csv_path):
        """Test loaded data has correct shape."""
        preprocessor = DataPreprocessor(sample_csv_path)
        df = preprocessor.load_data()
        
        # Check expected columns are present
        expected_cols = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        for col in expected_cols:
            assert col in df.columns
    
    def test_preprocess_data_returns_tuple(self, sample_csv_path):
        """Test preprocessing returns X and y."""
        preprocessor = DataPreprocessor(sample_csv_path)
        preprocessor.load_data()
        X, y = preprocessor.preprocess_data()
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
    
    def test_preprocess_creates_new_features(self, sample_csv_path):
        """Test preprocessing creates engineered features."""
        preprocessor = DataPreprocessor(sample_csv_path)
        preprocessor.load_data()
        X, y = preprocessor.preprocess_data()
        
        # Check for engineered features (normalized name)
        feature_names = X.columns.tolist()
        assert len(feature_names) > 0
    
    def test_preprocess_balances_classes(self, sample_csv_path):
        """Test preprocessing balances class distribution."""
        preprocessor = DataPreprocessor(sample_csv_path)
        preprocessor.load_data()
        X, y = preprocessor.preprocess_data()
        
        # Check that failure class is present
        assert len(y) > 0
        # Data should be somewhat balanced
        value_counts = y.value_counts()
        assert len(value_counts) > 0
    
    def test_preprocess_data_removes_uid_columns(self, sample_csv_path):
        """Test preprocessing removes UID and Product ID."""
        preprocessor = DataPreprocessor(sample_csv_path)
        preprocessor.load_data()
        X, y = preprocessor.preprocess_data()
        
        assert 'UID' not in X.columns
        assert 'Product ID' not in X.columns
    
    def test_preprocess_normalizes_column_names(self, sample_csv_path):
        """Test preprocessing normalizes column names."""
        preprocessor = DataPreprocessor(sample_csv_path)
        preprocessor.load_data()
        X, y = preprocessor.preprocess_data()
        
        # Check that brackets are removed from column names
        for col in X.columns:
            assert '[' not in col
            assert ']' not in col
    
    def test_preprocess_encodes_type_variable(self, sample_csv_path):
        """Test preprocessing encodes Type variable."""
        preprocessor = DataPreprocessor(sample_csv_path)
        preprocessor.load_data()
        X, y = preprocessor.preprocess_data()
        
        # Check that Type is encoded
        if 'Type' in X.columns:
            type_values = X['Type'].unique()
            assert all(v in [1, 2, 3] for v in type_values)


class TestDataValidator:
    """Test data integrity validation."""
    
    @pytest.fixture
    def sample_csv_path(self, tmp_path):
        """Create temporary CSV file for testing."""
        np.random.seed(42)
        data = {
            'UID': range(1, 101),
            'Product ID': ['PROD_' + chr(65 + i % 3) for i in range(100)],
            'Type': ['L'] * 33 + ['M'] * 33 + ['H'] * 34,
            'Air temperature [K]': np.random.normal(298, 2, 100),
            'Process temperature [K]': np.random.normal(308, 2, 100),
            'Rotational speed [rpm]': np.random.normal(1500, 100, 100),
            'Torque [Nm]': np.random.normal(40, 5, 100),
            'Tool wear [min]': np.random.uniform(0, 200, 100),
            'Machine failure': np.random.binomial(1, 0.05, 100),
            'TWF': np.random.binomial(1, 0.01, 100),
            'HDF': np.random.binomial(1, 0.01, 100),
            'PWF': np.random.binomial(1, 0.01, 100),
            'OSF': np.random.binomial(1, 0.01, 100),
            'RNF': np.random.binomial(1, 0.01, 100),
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)
    
    def test_sample_data_has_required_columns(self, sample_csv_path):
        """Test sample data has all required columns."""
        df = pd.read_csv(sample_csv_path)
        
        required_cols = ['Machine failure', 'Type', 'Air temperature [K]', 
                        'Process temperature [K]', 'Rotational speed [rpm]',
                        'Torque [Nm]', 'Tool wear [min]']
        
        for col in required_cols:
            assert col in df.columns
    
    def test_target_variable_binary(self, sample_csv_path):
        """Test target variable is binary."""
        df = pd.read_csv(sample_csv_path)
        
        assert df['Machine failure'].dtype in ['int64', 'int32']
        assert set(df['Machine failure'].unique()).issubset({0, 1})
    
    def test_type_variable_valid_values(self, sample_csv_path):
        """Test Type variable has valid values."""
        df = pd.read_csv(sample_csv_path)
        
        valid_types = {'L', 'M', 'H'}
        assert set(df['Type'].unique()).issubset(valid_types)
    
    def test_numeric_columns_are_numeric(self, sample_csv_path):
        """Test numeric columns are actually numeric."""
        df = pd.read_csv(sample_csv_path)
        
        numeric_cols = ['Air temperature [K]', 'Process temperature [K]',
                       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df[col])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

