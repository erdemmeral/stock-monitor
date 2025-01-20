import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.earnings_estimator import EarningsEstimator

class TestEarningsEstimator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.estimator = EarningsEstimator()
        
        # Create sample historical EPS data
        dates = pd.date_range(start='2023-01-01', periods=8, freq='QE')
        self.sample_eps_data = pd.DataFrame({
            'Earnings': [1.0, 1.2, 0.8, 1.5, 1.1, 1.3, 0.9, 1.6],
            'Quarter': [d.quarter for d in dates],
            'Year': [d.year for d in dates]
        }, index=dates)

    def test_calculate_growth_rates(self):
        """Test growth rate calculations"""
        growth_rates = self.estimator.calculate_growth_rates(self.sample_eps_data)
        
        # Test QoQ growth
        latest = self.sample_eps_data['Earnings'].iloc[-1]
        previous = self.sample_eps_data['Earnings'].iloc[-2]
        expected_qoq = ((latest - previous) / abs(previous)) * 100
        self.assertAlmostEqual(growth_rates['qoq_growth'], expected_qoq, places=1)
        
        # Test YoY growth
        year_ago = self.sample_eps_data['Earnings'].iloc[-4]
        expected_yoy = ((latest - year_ago) / abs(year_ago)) * 100
        self.assertAlmostEqual(growth_rates['yoy_growth'], expected_yoy, places=1)
        
        # Test extreme values handling
        extreme_data = self.sample_eps_data.copy()
        extreme_data.loc[extreme_data.index[-1], 'Earnings'] = 100.0
        growth_rates = self.estimator.calculate_growth_rates(extreme_data)
        self.assertLessEqual(growth_rates['qoq_growth'], 1000)

    def test_get_quarter_name(self):
        """Test quarter name formatting"""
        # Test Q1
        q1_date = pd.Timestamp('2024-01-01')
        q1_name = self.estimator.get_quarter_name(q1_date)
        self.assertIn('Q2 2024', q1_name)
        self.assertIn('Expected: April', q1_name)
        
        # Test Q4
        q4_date = pd.Timestamp('2023-12-31')
        q4_name = self.estimator.get_quarter_name(q4_date)
        self.assertIn('Q1 2024', q4_name)
        self.assertIn('Expected: January', q4_name)

    @patch('yfinance.Ticker')
    def test_get_historical_eps_data(self, mock_ticker):
        """Test fetching historical EPS data"""
        # Mock yfinance response
        mock_stock = Mock()
        
        # Create mock quarterly financials with proper structure
        dates = pd.date_range(start='2023-01-01', periods=3, freq='QE')
        mock_financials = pd.DataFrame(
            data={
                'Net Income': [1000000, 1200000, 800000]
            },
            index=dates
        )
        mock_financials = mock_financials.T  # Transpose to match yfinance format
        
        mock_stock.quarterly_financials = mock_financials
        mock_stock.info = {'sharesOutstanding': 1000000}
        
        # Set up the mock to return our mock_stock
        mock_ticker.return_value = mock_stock
        
        # Test with valid data
        eps_data = self.estimator.get_historical_eps_data('AAPL')
        
        self.assertFalse(eps_data.empty)
        self.assertTrue('Earnings' in eps_data.columns)
        self.assertTrue('Quarter' in eps_data.columns)
        self.assertTrue('Year' in eps_data.columns)
        
        # Test calculations
        expected_eps = 1.0  # 1000000 / 1000000
        self.assertAlmostEqual(eps_data['Earnings'].iloc[0], expected_eps, places=2)

    def test_estimate_next_quarter_eps(self):
        """Test EPS estimation"""
        with patch.object(EarningsEstimator, 'get_historical_eps_data') as mock_hist, \
             patch.object(EarningsEstimator, 'get_peer_analysis') as mock_peer, \
             patch.object(EarningsEstimator, 'get_financial_ratios') as mock_ratios:
            
            mock_hist.return_value = self.sample_eps_data
            mock_peer.return_value = {'avg_qoq_growth': 5.0, 'avg_yoy_growth': 10.0}
            mock_ratios.return_value = {
                'gross_margin': 40.0,
                'operating_margin': 30.0,
                'net_margin': 20.0
            }
            
            estimated_eps, confidence = self.estimator.estimate_next_quarter_eps('AAPL')
            
            self.assertGreater(estimated_eps, 0)
            self.assertLessEqual(confidence['overall_confidence'], 1.0)
            self.assertGreaterEqual(confidence['overall_confidence'], 0.0)

    def test_format_estimation_report(self):
        """Test report formatting"""
        with patch.object(EarningsEstimator, 'get_historical_eps_data') as mock_hist, \
             patch.object(EarningsEstimator, 'estimate_next_quarter_eps') as mock_est, \
             patch.object(EarningsEstimator, 'get_peer_analysis') as mock_peer, \
             patch.object(EarningsEstimator, 'get_financial_ratios') as mock_ratios:
            
            mock_hist.return_value = self.sample_eps_data
            mock_est.return_value = (1.7, {
                'historical_volatility': 0.2,
                'peer_alignment': 0.8,
                'growth_stability': 0.7,
                'margin_stability': 0.6,
                'overall_confidence': 0.65
            })
            mock_peer.return_value = {'avg_qoq_growth': 5.0, 'avg_yoy_growth': 10.0}
            mock_ratios.return_value = {
                'gross_margin': 40.0,
                'operating_margin': 30.0,
                'net_margin': 20.0
            }
            
            report = self.estimator.format_estimation_report('AAPL')
            
            self.assertIn('AAPL', report)
            self.assertIn('Historical Quarterly EPS:', report)
            self.assertIn('Estimated EPS', report)
            self.assertIn('Growth Metrics:', report)
            self.assertIn('Confidence Metrics Explained:', report)

    def test_error_handling(self):
        """Test error handling for invalid data"""
        # Test with empty DataFrame
        growth_rates = self.estimator.calculate_growth_rates(pd.DataFrame())
        self.assertEqual(growth_rates['qoq_growth'], 0.0)
        
        # Test with NaN values
        data_with_nan = self.sample_eps_data.copy()
        data_with_nan.loc[data_with_nan.index[-1], 'Earnings'] = np.nan
        growth_rates = self.estimator.calculate_growth_rates(data_with_nan)
        self.assertIsInstance(growth_rates['qoq_growth'], float)

if __name__ == '__main__':
    unittest.main() 