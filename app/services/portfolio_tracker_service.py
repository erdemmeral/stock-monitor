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
                logger.info(f"Attempt {attempt + 1}/{self.max_retries}")
                logger.info(f"Making {method} request to {self.base_url}/{endpoint}")
                logger.info(f"Request data: {data}")
                if data and ('entryDate' in data or 'targetDate' in data):
                    logger.info(f"Date fields - Entry: {data.get('entryDate', 'N/A')}, Target: {data.get('targetDate', 'N/A')}")
                    logger.info(f"Date types - Entry: {type(data.get('entryDate', None))}, Target: {type(data.get('targetDate', None))}")
                
                async with self.session.request(
                    method=method,
                    url=f"{self.base_url}/{endpoint}",
                    json=data,
                    timeout=self.timeout,
                    ssl=False,  # Disable SSL verification
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    response_text = await response.text()
                    logger.info(f"Response received - Status: {response.status}")
                    logger.info(f"Response headers: {dict(response.headers)}")
                    logger.info(f"Response text: {response_text}")
                    
                    if response.status in [200, 201]:
                        try:
                            json_response = await response.json()
                            logger.info(f"Successfully parsed JSON response: {json_response}")
                            return json_response
                        except Exception as json_error:
                            logger.error(f"Error parsing JSON response: {str(json_error)}")
                            logger.error(f"Raw response: {response_text}")
                            return None
                    else:
                        logger.error(f"Request failed with status {response.status}")
                        logger.error(f"Response text: {response_text}")
                        logger.error(f"Request URL: {self.base_url}/{endpoint}")
                        logger.error(f"Request method: {method}")
                        logger.error(f"Request data: {data}")
                        return None
            except asyncio.TimeoutError as e:
                logger.error(f"Timeout on attempt {attempt + 1}/{self.max_retries}")
                logger.error(f"Request URL: {self.base_url}/{endpoint}")
                logger.error(f"Request method: {method}")
                logger.error(f"Request data: {data}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Waiting {self.retry_delay} seconds before retry...")
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(f"Connection timeout to host {self.base_url}/{endpoint}")
                    raise

            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}/{self.max_retries}: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Request URL: {self.base_url}/{endpoint}")
                logger.error(f"Request method: {method}")
                logger.error(f"Request data: {data}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Waiting {self.retry_delay} seconds before retry...")
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
            logger.info(f"Sending buy signal to tracker: {data}")
            logger.info(f"Date fields - Entry: {entry_date}, Target: {target_date}")
            logger.info(f"Date types - Entry: {type(entry_date)}, Target: {type(target_date)}")
            
            result = await self._make_request("POST", "positions", data)
            
            if result is not None:
                logger.info(f"Successfully sent buy signal for {symbol}")
                logger.info(f"Response data: {result}")
                return True
            else:
                logger.error(f"Failed to send buy signal for {symbol}")
                logger.error(f"Request data: {data}")
                return False
                
        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"Error type: {error_type}")
            logger.error(f"Error message: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            logger.error(f"Failed request data: {data}")
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