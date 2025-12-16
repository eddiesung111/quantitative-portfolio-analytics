import pytest
import numpy as np
from src.portfolio_optimizer import optimize_portfolio

# --- Fixture (Replaces 'setUp') ---
@pytest.fixture
def optimizer_data():
    """
    Creates the dummy data for use in all tests.
    Returns a tuple: (mean_returns, cov_matrix, risk_free_rate)
    """
    mean_returns = np.array([0.1, 0.2, 0.15])
    cov_matrix = np.array([
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1]
    ])
    risk_free_rate = 0.04
    
    return mean_returns, cov_matrix, risk_free_rate

# ---------------------------------
#           UNIT TESTS
# ---------------------------------

def test_weights_sum_to_one(optimizer_data):
    """
    Test if the optimized weights sum to exactly 1.0
    """
    means, cov, rf = optimizer_data
    result = optimize_portfolio(means, cov, rf)
    weights = result.x

    assert np.sum(weights) == pytest.approx(1.0)

def test_no_short_selling(optimizer_data):
    """
    Test if all weights are non-negative (>= 0)
    """
    means, cov, rf = optimizer_data
    
    result = optimize_portfolio(means, cov, rf)
    weights = result.x

    assert np.all(weights >= -1e-4)

def test_optimization_success(optimizer_data):
    """
    Check if the solver actually reports 'Success'
    """
    means, cov, rf = optimizer_data
    result = optimize_portfolio(means, cov, rf)

    assert result.success