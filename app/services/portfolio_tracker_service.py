import aiohttp
import logging
from typing import Optional
from datetime import datetime, timedelta
import pytz

logger = logging.getLogger(__name__)

class PortfolioTrackerService:
    def __init__(self):
        self.base_url = "https://portfolio-tracker-rough-dawn-5271.fly.dev"
        self.session = None
        
    async def _ensure_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None
            
    async def send_buy_signal(self, symbol: str, entry_price: float, target_price: float, entry_date: str, target_date: str):
        """Send buy signal to portfolio tracker"""
        await self._ensure_session()
        
        data = {
            "symbol": symbol.upper(),
            "entryPrice": float(entry_price),
            "targetPrice": float(target_price),
            "entryDate": entry_date,
            "targetDate": target_date
        }
        
        try:
            logger.info(f"Sending buy signal to portfolio tracker: {data}")
            async with self.session.post(f"{self.base_url}/api/positions", json=data) as response:
                response_text = await response.text()
                logger.info(f"Portfolio tracker response: {response.status} - {response_text}")
                
                if response.status == 201:
                    logger.info(f"Buy signal sent successfully for {symbol}")
                    return True
                else:
                    logger.error(f"Failed to send buy signal for {symbol}. Status: {response.status}, Response: {response_text}")
                    return False
        except Exception as e:
            logger.error(f"Error sending buy signal for {symbol}: {str(e)}")
            return False
            
    async def send_sell_signal(self, symbol: str, selling_price: float):
        """Send sell signal to portfolio tracker"""
        await self._ensure_session()
        
        data = {
            "soldPrice": selling_price
        }
        
        try:
            async with self.session.post(f"{self.base_url}/api/positions/{symbol}/sell", json=data) as response:
                if response.status == 200:
                    logger.info(f"Sell signal sent for {symbol}")
                    return await response.json()
                else:
                    logger.error(f"Failed to send sell signal for {symbol}. Status: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error sending sell signal for {symbol}: {str(e)}")
            return None

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