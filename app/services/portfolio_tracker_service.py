import aiohttp
import logging
from typing import Optional, List, Dict
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
        self.base_url = "https://portfolio-tracker-rough-dawn-5271.fly.dev/api"
        self.session = None
        self.max_retries = 3
        self.timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.retry_delay = 5  # seconds between retries

    async def _init_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)

    async def _make_request(self, method: str, endpoint: str, data: dict = None) -> Optional[dict]:
        """Make HTTP request with retry logic"""
        await self._init_session()
        
        for attempt in range(self.max_retries):
            try:
                async with self.session.request(
                    method=method,
                    url=f"{self.base_url}/{endpoint}",
                    json=data,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Request failed with status {response.status}: {await response.text()}")
                        return None

            except asyncio.TimeoutError as e:
                logger.error(f"Timeout on attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(f"Connection timeout to host {self.base_url}/{endpoint}")
                    raise

            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}/{self.max_retries}: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise

        return None

    async def send_buy_signal(self, symbol: str, entry_price: float, target_price: float,
                            entry_date: str, target_date: str) -> bool:
        """Send buy signal to portfolio tracker"""
        try:
            data = {
                "symbol": symbol,
                "entryPrice": float(entry_price),
                "targetPrice": float(target_price),
                "entryDate": entry_date,
                "targetDate": target_date
            }
            
            result = await self._make_request("POST", "positions", data)
            success = result is not None
            
            if success:
                logger.info(f"Successfully sent buy signal for {symbol}")
            else:
                logger.error(f"Failed to send buy signal for {symbol}")
            
            return success

        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"Error type: {error_type}")
            logger.error(f"Error message: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            logger.error(f"Unexpected error in send_buy_signal for {symbol}")
            return False

    async def send_sell_signal(self, symbol: str, selling_price: float, sell_condition: str = None) -> bool:
        """Send sell signal to portfolio tracker"""
        try:
            data = {
                "symbol": symbol,
                "selling_price": selling_price,
                "sell_condition": sell_condition
            }
            
            result = await self._make_request("POST", f"positions/{symbol}/sell", data)
            success = result is not None
            
            if success:
                logger.info(f"Successfully sent sell signal for {symbol}")
            else:
                logger.error(f"Failed to send sell signal for {symbol}")
            
            return success

        except Exception as e:
            logger.error(f"Unexpected error in send_sell_signal for {symbol}: {str(e)}")
            return False

    async def get_positions(self) -> Optional[List[Dict]]:
        """Get current positions from portfolio tracker"""
        try:
            result = await self._make_request("GET", "positions")
            return result if result else []
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return []

    async def close(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()

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