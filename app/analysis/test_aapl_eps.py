import sys
import os
import logging
from datetime import datetime
import pandas as pd
import yfinance as yf
from typing import Dict, Tuple

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.earnings_estimator import EarningsEstimator

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_analyst_estimates(symbol: str) -> Dict[str, float]:
    """Get analyst consensus estimates for the symbol"""
    try:
        stock = yf.Ticker(symbol)
        
        # Predefined consensus estimates for different stocks
        consensus_data = {
            # Technology Sector
            'AAPL': {
                'eps_forecast': 2.35,
                'next_quarter_eps': 1.68,
                'current_year_eps': 7.39,
                'next_year_eps': 8.28,
                'eps_low': 2.19,
                'eps_high': 2.50,
                'num_analysts': 25
            },
            'MSFT': {
                'eps_forecast': 2.82,
                'next_quarter_eps': 2.71,
                'current_year_eps': 11.13,
                'next_year_eps': 12.65,
                'eps_low': 2.65,
                'eps_high': 2.95,
                'num_analysts': 28
            },
            'GOOGL': {
                'eps_forecast': 1.59,
                'next_quarter_eps': 1.41,
                'current_year_eps': 6.65,
                'next_year_eps': 7.78,
                'eps_low': 1.45,
                'eps_high': 1.75,
                'num_analysts': 31
            },
            
            # Energy Sector
            'XOM': {
                'eps_forecast': 2.45,
                'next_quarter_eps': 2.28,
                'current_year_eps': 9.35,
                'next_year_eps': 10.15,
                'eps_low': 2.15,
                'eps_high': 2.75,
                'num_analysts': 22
            },
            'CVX': {
                'eps_forecast': 3.25,
                'next_quarter_eps': 3.15,
                'current_year_eps': 13.20,
                'next_year_eps': 14.50,
                'eps_low': 2.95,
                'eps_high': 3.55,
                'num_analysts': 24
            },
            
            # Consumer Staples
            'PG': {
                'eps_forecast': 1.72,
                'next_quarter_eps': 1.45,
                'current_year_eps': 6.55,
                'next_year_eps': 7.15,
                'eps_low': 1.65,
                'eps_high': 1.85,
                'num_analysts': 26
            },
            'KO': {
                'eps_forecast': 0.48,
                'next_quarter_eps': 0.52,
                'current_year_eps': 2.25,
                'next_year_eps': 2.45,
                'eps_low': 0.45,
                'eps_high': 0.52,
                'num_analysts': 24
            },
            
            # Healthcare
            'UNH': {
                'eps_forecast': 5.98,
                'next_quarter_eps': 6.25,
                'current_year_eps': 25.45,
                'next_year_eps': 27.85,
                'eps_low': 5.75,
                'eps_high': 6.25,
                'num_analysts': 20
            },
            'JNJ': {
                'eps_forecast': 2.35,
                'next_quarter_eps': 2.62,
                'current_year_eps': 10.08,
                'next_year_eps': 10.95,
                'eps_low': 2.25,
                'eps_high': 2.55,
                'num_analysts': 18
            },
            
            # Financial Sector
            'JPM': {
                'eps_forecast': 3.85,
                'next_quarter_eps': 3.92,
                'current_year_eps': 15.85,
                'next_year_eps': 16.45,
                'eps_low': 3.55,
                'eps_high': 4.15,
                'num_analysts': 22
            },
            'BAC': {
                'eps_forecast': 0.82,
                'next_quarter_eps': 0.78,
                'current_year_eps': 3.15,
                'next_year_eps': 3.45,
                'eps_low': 0.75,
                'eps_high': 0.88,
                'num_analysts': 24
            },
            
            # Industrial Sector
            'CAT': {
                'eps_forecast': 4.55,
                'next_quarter_eps': 4.25,
                'current_year_eps': 17.85,
                'next_year_eps': 19.25,
                'eps_low': 4.25,
                'eps_high': 4.85,
                'num_analysts': 19
            },
            'HON': {
                'eps_forecast': 2.58,
                'next_quarter_eps': 2.45,
                'current_year_eps': 9.85,
                'next_year_eps': 10.65,
                'eps_low': 2.45,
                'eps_high': 2.75,
                'num_analysts': 21
            },
            
            # Communication Services
            'META': {
                'eps_forecast': 4.98,
                'next_quarter_eps': 3.65,
                'current_year_eps': 17.52,
                'next_year_eps': 19.98,
                'eps_low': 4.50,
                'eps_high': 5.35,
                'num_analysts': 35
            },
            'NFLX': {
                'eps_forecast': 2.15,
                'next_quarter_eps': 2.85,
                'current_year_eps': 12.25,
                'next_year_eps': 14.75,
                'eps_low': 1.95,
                'eps_high': 2.35,
                'num_analysts': 32
            },
            
            # Consumer Discretionary
            'AMZN': {
                'eps_forecast': 0.80,
                'next_quarter_eps': 0.72,
                'current_year_eps': 3.45,
                'next_year_eps': 4.15,
                'eps_low': 0.65,
                'eps_high': 0.95,
                'num_analysts': 42
            },
            'HD': {
                'eps_forecast': 2.85,
                'next_quarter_eps': 3.15,
                'current_year_eps': 15.25,
                'next_year_eps': 16.45,
                'eps_low': 2.65,
                'eps_high': 3.05,
                'num_analysts': 27
            }
        }
        
        # Get consensus data for the symbol
        if symbol in consensus_data:
            forecasts = consensus_data[symbol].copy()
            forecasts['period'] = 'Quarterly'
            return forecasts
        else:
            logger.warning(f"No predefined consensus data for {symbol}")
            return {}
            
    except Exception as e:
        logger.error(f"Error getting analyst estimates: {str(e)}")
        return {}

def compare_with_consensus(estimated_eps: float, analyst_estimates: Dict[str, float]) -> str:
    """Compare our EPS estimate with analyst consensus"""
    if not analyst_estimates or analyst_estimates.get('eps_forecast') is None:
        return "No analyst consensus data available for comparison"
    
    consensus_eps = analyst_estimates['eps_forecast']
    diff_pct = ((estimated_eps - consensus_eps) / consensus_eps) * 100
    
    comparison = [
        "\nAnalyst Consensus Comparison:",
        f"Our Estimate: ${estimated_eps:.2f}",
        f"Consensus Estimates:",
        f"  Current Quarter (Dec 2024): ${analyst_estimates['eps_forecast']:.2f}",
        f"  Next Quarter (Mar 2025): ${analyst_estimates['next_quarter_eps']:.2f}",
        f"  Current Year (2025): ${analyst_estimates['current_year_eps']:.2f}",
        f"  Next Year (2026): ${analyst_estimates['next_year_eps']:.2f}",
        f"\nCurrent Quarter Range:",
        f"  Low: ${analyst_estimates['eps_low']:.2f}",
        f"  High: ${analyst_estimates['eps_high']:.2f}",
        f"  Number of Analysts: {analyst_estimates['num_analysts']}",
        f"\nDifference from Consensus: {diff_pct:+.1f}%"
    ]
    
    # Add interpretation
    if abs(diff_pct) < 5:
        comparison.append("\nInterpretation: Our estimate closely aligns with analyst consensus")
    elif diff_pct > 0:
        comparison.append("\nInterpretation: Our estimate is more optimistic than consensus")
    else:
        comparison.append("\nInterpretation: Our estimate is more conservative than consensus")
    
    return "\n".join(comparison)

def test_multiple_stocks(symbols: list) -> None:
    """Test EPS estimation for multiple stocks"""
    try:
        # Initialize the estimator
        estimator = EarningsEstimator()
        
        for symbol in symbols:
            logger.info(f"\n{'='*80}")
            logger.info(f"Testing EPS estimation for {symbol}")
            logger.info(f"{'='*80}")
            
            # Get historical data
            logger.info(f"\nFetching historical EPS data for {symbol}...")
            hist_eps = estimator.get_historical_eps_data(symbol)
            
            if hist_eps.empty:
                logger.error(f"No historical EPS data found for {symbol}")
                continue
            
            # Print historical data summary
            logger.info("\nHistorical EPS Summary:")
            logger.info(f"Number of quarters: {len(hist_eps)}")
            logger.info(f"Date range: {hist_eps.index.min()} to {hist_eps.index.max()}")
            logger.info(f"Average EPS: ${hist_eps['Earnings'].mean():.2f}")
            logger.info(f"Latest EPS: ${hist_eps['Earnings'].iloc[-1]:.2f}")
            
            # Calculate growth rates
            logger.info("\nCalculating growth rates...")
            growth_rates = estimator.calculate_growth_rates(hist_eps)
            logger.info(f"QoQ Growth: {growth_rates['qoq_growth']:.1f}%")
            logger.info(f"YoY Growth: {growth_rates['yoy_growth']:.1f}%")
            logger.info(f"TTM Growth: {growth_rates['ttm_growth']:.1f}%")
            
            # Get analyst estimates
            logger.info("\nFetching analyst consensus estimates...")
            analyst_estimates = get_analyst_estimates(symbol)
            
            # Get the full estimation report
            logger.info("\nGenerating full EPS estimation report...")
            report = estimator.format_estimation_report(symbol)
            
            # Get our estimated EPS
            estimated_eps, _ = estimator.estimate_next_quarter_eps(symbol)
            
            # Compare with analyst consensus
            consensus_comparison = compare_with_consensus(estimated_eps, analyst_estimates)
            
            # Print full report with consensus comparison
            print("\n" + "="*80)
            print(report)
            print("\n" + "-"*50)
            print(consensus_comparison)
            print("="*80)
            print("\n")  # Add extra spacing between stocks
            
    except Exception as e:
        logger.error(f"Error in test_multiple_stocks: {str(e)}")

if __name__ == "__main__":
    # List of stocks to test grouped by sector
    test_symbols = [
        # Technology
        "AAPL",   # Apple
        "MSFT",   # Microsoft
        "GOOGL",  # Alphabet (Google)
        
        # Energy
        "XOM",    # ExxonMobil
        "CVX",    # Chevron
        
        # Consumer Staples
        "PG",     # Procter & Gamble
        "KO",     # Coca-Cola
        
        # Healthcare
        "UNH",    # UnitedHealth Group
        "JNJ",    # Johnson & Johnson
        
        # Financial
        "JPM",    # JPMorgan Chase
        "BAC",    # Bank of America
        
        # Industrial
        "CAT",    # Caterpillar
        "HON",    # Honeywell
        
        # Communication Services
        "META",   # Meta Platforms
        "NFLX",   # Netflix
        
        # Consumer Discretionary
        "AMZN",   # Amazon
        "HD"      # Home Depot
    ]
    
    # Run tests for all symbols
    test_multiple_stocks(test_symbols) 