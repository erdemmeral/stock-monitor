from earnings_estimator import EarningsEstimator
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # Initialize the estimator
    estimator = EarningsEstimator()
    
    # Test companies
    test_symbols = [
        "AAPL",  # Technology
        "AMZN",  # E-Commerce
        "JPM",   # Financial
        "JNJ"    # Healthcare
    ]
    
    print("\nRunning EPS Estimations...")
    print("=" * 70)
    
    for symbol in test_symbols:
        try:
            print(f"\nAnalyzing {symbol}...")
            report = estimator.format_estimation_report(symbol)
            print("\n" + report)
            print("\n" + "=" * 70)
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 