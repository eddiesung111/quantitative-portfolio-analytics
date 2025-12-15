import unittest
import numpy as np
from src.portfolio_optimizer import optimize_portfolio

class TestPortfolioOptimizer(unittest.TestCase):
    
    def setUp(self):
        """
        Set up a dummy scenario for testing.
        We create 3 fake assets with simple returns.
        """
        # 3 Assets
        self.mean_returns = np.array([0.1, 0.2, 0.15])
        
        # Simple Identity Matrix for Covariance (Uncorrelated)
        self.cov_matrix = np.array([
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1]
        ])
        
        self.risk_free_rate = 0.04

    def test_weights_sum_to_one(self):
        """
        Test if the optimized weights sum to exactly 1.0
        """
        result = optimize_portfolio(self.mean_returns, self.cov_matrix, self.risk_free_rate)
        weights = result.x

        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)

    def test_no_short_selling(self):
        """
        Test if all weights are non-negative (>= 0)
        """
        result = optimize_portfolio(self.mean_returns, self.cov_matrix, self.risk_free_rate)
        weights = result.x
        self.assertTrue(np.all(weights >= -1e-4))

    def test_optimization_success(self):
        """
        Check if the solver actually reports 'Success'
        """
        result = optimize_portfolio(self.mean_returns, self.cov_matrix, self.risk_free_rate)
        self.assertTrue(result.success)

if __name__ == '__main__':
    unittest.main()
