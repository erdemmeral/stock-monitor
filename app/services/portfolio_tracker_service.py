import aiohttp
import logging
from typing import Optional
from datetime import datetime, timedelta
import pytz

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
        logger.info("Initializing PortfolioTrackerService")
        self.base_url = "https://portfolio-tracker-rough-dawn-5271.fly.dev"
        self.session = None
        logger.info(f"Base URL set to: {self.base_url}")
        
    async def _ensure_session(self):
        """Ensure an active session exists"""
        try:
            if self.session is None:
                logger.info("Creating new aiohttp session")
                timeout = aiohttp.ClientTimeout(total=30)  # 30 seconds timeout
                self.session = aiohttp.ClientSession(timeout=timeout)
                logger.info("Session created successfully")
            elif self.session.closed:
                logger.info("Session was closed, creating new session")
                timeout = aiohttp.ClientTimeout(total=30)
                self.session = aiohttp.ClientSession(timeout=timeout)
                logger.info("Session recreated successfully")
        except Exception as e:
            logger.error(f"Error ensuring session: {str(e)}", exc_info=True)
            raise

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None
            
    async def send_buy_signal(self, symbol: str, entry_price: float, target_price: float, entry_date: str, target_date: str):
        """Send buy signal to portfolio tracker"""
        try:
            logger.info(f"=== Starting buy signal process for {symbol} ===")
            logger.info(f"Ensuring session is active for {symbol}")
            await self._ensure_session()
            
            # Format dates to match server expectations
            entry_date_formatted = datetime.fromisoformat(entry_date.replace('Z', '+00:00')).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            target_date_formatted = datetime.fromisoformat(target_date.replace('Z', '+00:00')).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            logger.info(f"Formatted dates - Entry: {entry_date_formatted}, Target: {target_date_formatted}")
            
            data = {
                "symbol": symbol.upper(),
                "entryPrice": float(entry_price),
                "targetPrice": float(target_price),
                "entryDate": entry_date_formatted,
                "targetDate": target_date_formatted
            }
            
            url = f"{self.base_url}/api/positions"
            logger.info(f"=== Making POST request ===")
            logger.info(f"URL: {url}")
            logger.info(f"Data: {data}")
            
            try:
                headers = {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
                logger.info(f"Headers: {headers}")
                
                # First try with SSL
                try:
                    async with self.session.post(url, json=data, headers=headers) as response:
                        response_text = await response.text()
                        logger.info(f"Response status: {response.status}")
                        logger.info(f"Response headers: {dict(response.headers)}")
                        logger.info(f"Response body: {response_text}")
                        
                        if response.status == 201:
                            logger.info(f"Buy signal successfully sent for {symbol}")
                            return True
                        else:
                            logger.error(f"Failed to send buy signal for {symbol}")
                            logger.error(f"Status: {response.status}")
                            logger.error(f"Response: {response_text}")
                            return False
                            
                except aiohttp.ClientSSLError as ssl_error:
                    logger.warning(f"SSL Error occurred, retrying without SSL: {ssl_error}")
                    # Retry without SSL
                    async with self.session.post(url, json=data, headers=headers, ssl=False) as response:
                        response_text = await response.text()
                        logger.info(f"(No SSL) Response status: {response.status}")
                        logger.info(f"(No SSL) Response headers: {dict(response.headers)}")
                        logger.info(f"(No SSL) Response body: {response_text}")
                        
                        if response.status == 201:
                            logger.info(f"Buy signal successfully sent for {symbol} (without SSL)")
                            return True
                        else:
                            logger.error(f"Failed to send buy signal for {symbol} (without SSL)")
                            logger.error(f"Status: {response.status}")
                            logger.error(f"Response: {response_text}")
                            return False
                        
            except aiohttp.ClientError as e:
                logger.error(f"HTTP error while sending buy signal for {symbol}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error message: {str(e)}")
                logger.error("Stack trace:", exc_info=True)
                return False
                
        except Exception as e:
            logger.error(f"Unexpected error in send_buy_signal for {symbol}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return False
            
    async def send_sell_signal(self, symbol: str, selling_price: float):
        """Send sell signal to portfolio tracker"""
        await self._ensure_session()
        
        try:
            # First, get all positions to find the one with matching symbol
            async with self.session.get(f"{self.base_url}/api/portfolio") as response:
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
                
                async with self.session.patch(f"{self.base_url}/api/positions/{position_id}/update-price", json=data) as sell_response:
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
        await self._ensure_session()
        try:
            async with self.session.get(f"{self.base_url}/api/portfolio") as response:
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
        await self._ensure_session()
        try:
            async with self.session.get(f"{self.base_url}/api/performance") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error("Failed to get performance metrics")
                    return None
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return None 