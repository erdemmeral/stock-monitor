import requests
import logging
import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def get_all_symbols() -> list:
    """Get list of stock symbols to monitor"""
    try:
        # Fetch stock list from GitHub repository
        url = "https://raw.githubusercontent.com/erdemmeral/stockslist/main/stock_tickers.txt"
        response = requests.get(url)
        response.raise_for_status()
        
        # Split the text into lines and remove empty lines
        symbols = [line.strip() for line in response.text.split('\n') if line.strip()]
        
        # Remove duplicates and sort
        symbols = sorted(list(set(symbols)))
        
        logger.info(f"Successfully fetched {len(symbols)} stock symbols")
        return symbols
        
    except Exception as e:
        logger.error(f"Error fetching stock symbols: {str(e)}")
        return []

def get_sp500_symbols() -> list:
    """Get list of S&P 500 symbols (kept for backward compatibility)"""
    return []

def get_nasdaq100_symbols() -> list:
    """Get list of NASDAQ 100 symbols (kept for backward compatibility)"""
    return [] 