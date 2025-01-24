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
        self.max_retries = 5  # Increased from 3
        self.timeout = aiohttp.ClientTimeout(
            total=60,     # Increased total timeout
            connect=20,   # Increased connect timeout
            sock_read=30  # Added socket read timeout
        )
        self.retry_delay = 10  # Increased from 5 seconds

    async def _init_session(self):
        """Initialize or reinitialize the session if needed"""
        try:
            if self.session is None or self.session.closed:
                logger.info("Creating new aiohttp session")
                self.session = aiohttp.ClientSession(timeout=self.timeout)
        except Exception as e:
            logger.error(f"Error initializing session: {str(e)}")
            if self.session and not self.session.closed:
                await self.session.close()
            self.session = aiohttp.ClientSession(timeout=self.timeout)

    async def _make_request(self, method: str, endpoint: str, data: dict = None) -> Optional[dict]:
        """Make HTTP request with retry logic"""
        for attempt in range(self.max_retries):
            try:
                # Ensure session is valid before each attempt
                await self._init_session()
                
                logger.info(f"Attempt {attempt + 1}/{self.max_retries}")
                logger.info(f"Making {method} request to {self.base_url}/{endpoint}")
                logger.info(f"Request data: {data}")
                
                async with self.session.request(
                    method=method,
                    url=f"{self.base_url}/{endpoint}",
                    json=data,
                    timeout=self.timeout,
                    ssl=False,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    response_text = await response.text()
                    logger.info(f"Response status: {response.status}")
                    logger.info(f"Response text: {response_text}")
                    
                    if response.status in [200, 201]:
                        try:
                            return await response.json()
                        except Exception as json_error:
                            logger.error(f"Error parsing JSON response: {str(json_error)}")
                            logger.error(f"Raw response: {response_text}")
                            return None
                    else:
                        logger.error(f"Request failed with status {response.status}")
                        logger.error(f"Response text: {response_text}")
                        if attempt < self.max_retries - 1:
                            logger.info(f"Waiting {self.retry_delay} seconds before retry...")
                            await asyncio.sleep(self.retry_delay)
                        return None

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"Connection error on attempt {attempt + 1}/{self.max_retries}: {str(e)}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Waiting {self.retry_delay} seconds before retry...")
                    await asyncio.sleep(self.retry_delay)
                    # Force session recreation on next attempt
                    if self.session and not self.session.closed:
                        await self.session.close()
                    self.session = None
                else:
                    raise

            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}/{self.max_retries}: {str(e)}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Waiting {self.retry_delay} seconds before retry...")
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise

        return None

    async def send_buy_signal(self, symbol: str, entry_price: float, target_price: float, 
                           entry_date: str, target_date: str) -> bool:
        """Send buy signal to portfolio tracker"""
        for attempt in range(self.max_retries):
            try:
                await self._init_session()
                
                data = {
                    "symbol": symbol,
                    "entryPrice": float(entry_price),
                    "targetPrice": float(target_price),
                    "entryDate": entry_date,
                    "targetDate": target_date
                }
                logger.info(f"Attempt {attempt + 1}/{self.max_retries} - Sending buy signal to tracker")
                logger.info(f"Request data: {data}")
                
                url = f"{self.base_url}/positions"
                async with self.session.post(url, json=data, ssl=False, headers={'Content-Type': 'application/json'}) as response:
                    response_text = await response.text()
                    logger.info(f"Response status: {response.status}")
                    logger.info(f"Response body: {response_text}")
                    
                    if response.status in (200, 201):
                        logger.info("Successfully sent buy signal")
                        return True
                    else:
                        logger.error(f"Failed to send buy signal. Status: {response.status}")
                        logger.error(f"Response: {response_text}")
                        if attempt < self.max_retries - 1:
                            logger.info(f"Waiting {self.retry_delay} seconds before retry...")
                            await asyncio.sleep(self.retry_delay)
                        return False
                    
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"Connection error on attempt {attempt + 1}: {str(e)}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Waiting {self.retry_delay} seconds before retry...")
                    await asyncio.sleep(self.retry_delay)
                    # Force session recreation
                    if self.session and not self.session.closed:
                        await self.session.close()
                    self.session = None
                else:
                    logger.error("Max retries reached")
                    return False
                    
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}", exc_info=True)
                return False
        
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

    async def test_buy_signal(self):
        """Test function to verify buy signal functionality"""
        test_data = {
            "symbol": "AIMD",
            "entryPrice": 0.7630000114440918,
            "targetPrice": 0.9266609814637333,
            "entryDate": "2025-01-21T15:00:00+00:00",
            "targetDate": "2025-02-20T15:00:00+00:00"
        }
        
        logger.info("=== Starting Buy Signal Test ===")
        logger.info(f"Test data: {test_data}")
        
        try:
            success = await self.send_buy_signal(
                symbol=test_data["symbol"],
                entry_price=test_data["entryPrice"],
                target_price=test_data["targetPrice"],
                entry_date=test_data["entryDate"],
                target_date=test_data["targetDate"]
            )
            
            if success:
                logger.info("✅ Buy signal test successful")
            else:
                logger.error("❌ Buy signal test failed")
                
        except Exception as e:
            logger.error(f"Test failed with error: {str(e)}", exc_info=True)
        
        logger.info("=== Buy Signal Test Complete ===")

async def run_test():
    """Run the portfolio tracker test"""
    service = PortfolioTrackerService()
    await service.test_buy_signal()
    await service.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_test()) 