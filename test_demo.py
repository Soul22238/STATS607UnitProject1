import pytest

import numpy as np
import pandas as pd

from demo import calculate_correlation, bootstrap_ci, robust_regression_analysis

class TestCorrelation:
    """Test suite"""
    
    def test_correlation_happy_path(self):
        """Test basic functionality - the docstring example."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        result = calculate_correlation(x, y)
        assert abs(result - 1.0) < 1e-10  # Perfect correlation
    
    def test_correlation_edge_cases(self):
        """Test edge cases that commonly cause bugs."""
        # Single value arrays
        with pytest.raises(ValueError, match="at least 2 observations"):
            calculate_correlation([1], [2])
        
        # Empty arrays
        with pytest.raises(ValueError, match="at least 2 observations"):
            calculate_correlation([], [])
        
        # Mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            calculate_correlation([1, 2, 3], [1, 2])
    
    def test_correlation_numerical_edge_cases(self):
        """Test numerical stability issues."""
        # Zero variance
        with pytest.raises(ValueError, match="zero variance"):
            calculate_correlation([1, 1, 1], [1, 2, 3])
        
        # Missing values
        with pytest.raises(ValueError, match="missing values"):
            calculate_correlation([1, 2, np.nan], [1, 2, 3])
    
    def test_correlation_types(self):
        """Test input type validation."""
        with pytest.raises(TypeError, match="arrays or lists"):
            calculate_correlation("invalid", [1, 2, 3])

def test_bootstrap_warnings():
    """Test that warnings are issued appropriately."""
    small_data = [1, 2, 3]  # Very small sample
    
    # Test warning about small sample size
    with pytest.warns(UserWarning, match="Small sample size"):
        bootstrap_ci(small_data, np.mean)
    
    # Test warning about few bootstrap samples
    with pytest.warns(UserWarning, match="bootstrap samples"):
        bootstrap_ci([1, 2, 3, 4, 5] * 10, np.mean, n_bootstrap=100)

def test_stratified_analysis_integration():
    """Integration test - split into parts"""
    # Test the main functionality without worrying about warnings
    np.random.seed(42)
    data = pd.DataFrame({
        'y': np.random.normal(0, 1, 100),
        'x': np.random.normal(0, 1, 100),
        'group': np.random.choice(['A', 'B', 'C'], 100, p=[0.5, 0.4, 0.1])
    })
    
    results = robust_regression_analysis(data, 'group', 'y', 'x')
    assert len(results) >= 2
    
    for group, model in results.items():
        assert hasattr(model, 'params'), f"Group {group} missing regression coefficients"

def test_small_group_warnings():
    """Test that warnings are issued for small groups"""
    # Force small groups
    data = pd.DataFrame({
        'y': [1, 2, 3],
        'x': [1, 2, 3],
        'group': ['A', 'B', 'C']  # Each group has only 1 observation
    })
    
    with pytest.warns(UserWarning, match="only.*observations"):
        robust_regression_analysis(data, 'group', 'y', 'x')