import asyncio
import sys
import os
import warnings

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Filter out yfinance warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from app.backtesting import run_backtest

if __name__ == "__main__":
    try:
        asyncio.run(run_backtest())
    except KeyboardInterrupt:
        print("\nBacktest interrupted by user")
    except Exception as e:
        print(f"\nError during backtest: {str(e)}")
    finally:
        print("\nBacktest completed") 