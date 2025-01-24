import aiohttp
import logging
from typing import Optional
from datetime import datetime, timedelta
import pytz
import asyncio
from time import sleep

# Configure logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add a stream handler if not already added
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class PortfolioTrackerService:
    def __init__(self):
        self.base_url = "https://portfolio-tracker-rough-dawn-5271.fly.dev"
        self.timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
    async def send_buy_signal(self, symbol: str, entry_price: float, target_price: float, 
                            entry_date: str, target_date: str) -> bool:
        """Send buy signal to portfolio tracker using a fresh session"""
        try:
            # Create a new session with increased timeout
            timeout = aiohttp.ClientTimeout(total=60, connect=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                data = {
                    "symbol": symbol,
                    "entryPrice": float(entry_price),
                    "targetPrice": float(target_price),
                    "entryDate": entry_date,
                    "targetDate": target_date
                }
                
                # Try up to 3 times with exponential backoff
                for attempt in range(3):
                    try:
                        async with session.post(f"{self.base_url}/api/positions", json=data, ssl=False) as response:
                            if response.status == 201:
                                logger.info(f"Successfully sent buy signal for {symbol}")
                                return True
                            else:
                                error_text = await response.text()
                                logger.error(f"Failed to send buy signal for {symbol}. Status: {response.status}, Response: {error_text}")
                                return False
                    except TimeoutError as e:
                        if attempt < 2:  # Don't sleep on last attempt
                            sleep_time = (2 ** attempt) * 5  # 5s, 10s
                            logger.warning(f"Timeout on attempt {attempt + 1}, retrying in {sleep_time}s...")
                            await asyncio.sleep(sleep_time)
                        else:
                            raise  # Re-raise on final attempt
                    
        except Exception as e:
            logger.error(f"Unexpected error in send_buy_signal for {symbol}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return False

    async def send_sell_signal(self, symbol: str, selling_price: float):
        """Send sell signal to portfolio tracker"""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                # First, get all positions to find the one with matching symbol
                async with session.get(f"{self.base_url}/api/portfolio") as response:
                    if response.status != 200:
                        logger.error(f"Failed to get portfolio for sell signal. Status: {response.status}")
                        return False
                        
                    portfolio = await response.json()
                    position = next((pos for pos in portfolio if pos['symbol'] == symbol.upper() and pos['status'] == 'OPEN'), None)
                    
                    if not position:
                        logger.error(f"No open position found for {symbol}")
                        return False
                        
                    position_id = position['_id']
                    
                    # Now send the sell signal with the position ID
                    data = {
                        "soldPrice": float(selling_price),
                        "status": "CLOSED"
                    }
                    
                    async with session.patch(f"{self.base_url}/api/positions/{position_id}/update-price", json=data) as sell_response:
                        if sell_response.status == 200:
                            logger.info(f"Sell signal sent successfully for {symbol}")
                            return True
                        else:
                            logger.error(f"Failed to send sell signal for {symbol}. Status: {sell_response.status}")
                            return False
                            
            except Exception as e:
                logger.error(f"Error sending sell signal for {symbol}: {str(e)}")
                return False

    async def get_portfolio_status(self):
        """Get current portfolio status"""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.get(f"{self.base_url}/api/portfolio") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error("Failed to get portfolio status")
                        return None
            except Exception as e:
                logger.error(f"Error getting portfolio status: {str(e)}")
                return None

    async def get_performance_metrics(self):
        """Get performance metrics"""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.get(f"{self.base_url}/api/performance") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error("Failed to get performance metrics")
                        return None
            except Exception as e:
                logger.error(f"Error getting performance metrics: {str(e)}")
                return None 