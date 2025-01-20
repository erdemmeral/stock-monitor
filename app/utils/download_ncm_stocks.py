import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_ncm_stocks():
    """Download list of NASDAQ Capital Market stocks"""
    try:
        base_url = "https://www.profitspi.com/stock/view.aspx"
        stocks = []
        page = 1
        total_pages = 39  # From website info: "Page 1 of 39. Rows 1 to 50 of 1921"
        
        # Set headers to mimic browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        while page <= total_pages:
            # Construct URL with pagination parameters
            url = f"{base_url}?v=exchange-symbols&p=NCM&pg={page}"
            logger.info(f"Downloading page {page} of {total_pages}")
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all stock rows
            rows = soup.find_all('tr')
            
            page_stocks = []
            for row in rows:
                # Find symbol cells
                symbol_cell = row.find('a')
                if symbol_cell and not symbol_cell.text.endswith('W'):  # Skip warrants
                    symbol = symbol_cell.text.strip()
                    if symbol and len(symbol) <= 5:  # Basic validation
                        page_stocks.append(symbol)
            
            if page_stocks:
                stocks.extend(page_stocks)
                logger.info(f"Found {len(page_stocks)} stocks on page {page}")
            else:
                logger.warning(f"No stocks found on page {page}")
            
            # Add delay between requests to be polite
            time.sleep(1)
            page += 1
        
        # Remove duplicates and sort
        stocks = sorted(list(set(stocks)))
        
        # Save to file
        with open('ncm_stocks.txt', 'w') as f:
            for symbol in stocks:
                f.write(f"{symbol}\n")
        
        logger.info(f"Successfully downloaded {len(stocks)} NCM stock tickers")
        return stocks
        
    except Exception as e:
        logger.error(f"Error downloading NCM stocks: {str(e)}")
        return []

if __name__ == "__main__":
    download_ncm_stocks() 