import asyncio
from datetime import datetime, timedelta, timezone
from .backtest_engine import BacktestEngine

async def main():
    try:
        # Use recent dates
        end_date = datetime(2024, 3, 22, 16, 0, tzinfo=timezone.utc)  # Recent trading day
        start_date = end_date - timedelta(days=90)  # 3 months before
        
        print("Initializing backtest...")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        
        # Initialize backtesting engine
        engine = BacktestEngine(start_date, end_date)
        
        # Use a smaller set of test stocks
        test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        print(f"\nTesting {len(test_symbols)} stocks: {', '.join(test_symbols)}")
        
        print("\nStarting sentiment analysis and backtesting...")
        print("This may take a few minutes...")
        
        # Run backtest
        result = await engine.run_backtest(test_symbols)
        
        # Print results
        engine.print_results()
        
    except Exception as e:
        print(f"\nError during backtest execution: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 