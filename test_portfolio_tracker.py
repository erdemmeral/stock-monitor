import asyncio
import logging
from datetime import datetime, timedelta
import pytz
from app.services.portfolio_tracker_service import PortfolioTrackerService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_buy_signal():
    try:
        logger.info("Starting portfolio tracker test")
        
        # Initialize service
        service = PortfolioTrackerService()
        logger.info("Portfolio tracker service initialized")
        
        # Test data
        symbol = "AAPL"
        entry_price = 180.50
        target_price = 200.00
        entry_date = datetime.now(tz=pytz.UTC)
        target_date = entry_date + timedelta(days=30)
        
        logger.info(f"Sending test buy signal for {symbol}")
        logger.info(f"Entry Price: ${entry_price}")
        logger.info(f"Target Price: ${target_price}")
        logger.info(f"Entry Date: {entry_date.isoformat()}")
        logger.info(f"Target Date: {target_date.isoformat()}")
        
        # Send buy signal
        success = await service.send_buy_signal(
            symbol=symbol,
            entry_price=entry_price,
            target_price=target_price,
            entry_date=entry_date.isoformat(),
            target_date=target_date.isoformat()
        )
        
        if success:
            logger.info("✅ Buy signal sent successfully")
        else:
            logger.error("❌ Failed to send buy signal")
            
        # Close the session
        await service.close()
        logger.info("Test completed")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        
async def main():
    await test_buy_signal()

if __name__ == "__main__":
    asyncio.run(main()) 