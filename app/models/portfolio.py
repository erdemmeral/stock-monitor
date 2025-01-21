from requests_cache import logger
from app.models.position import Position
class Portfolio:
    def __init__(self):
        self.positions = {}  # symbol -> Position

    def add_position(self, symbol, entry_price, current_price, target_price, entry_date, target_date, timeframe):
        try:
            logger.info(f"🔔 Attempting to add position for {symbol}")
            logger.info(f"💰 Entry Price: ${entry_price:.2f}")
            logger.info(f"🎯 Target Price: ${target_price:.2f}")
            logger.info(f"⏰ Target Date: {target_date}")
            logger.info(f"⏱ Timeframe: {timeframe}")

            # Create new position with all required arguments
            new_position = Position(
                symbol=symbol,
                entry_price=entry_price,
                current_price=current_price,
                target_price=target_price,
                entry_date=entry_date,
                target_date=target_date,
                timeframe=timeframe
            )

            # Add to positions
            self.positions[symbol] = new_position
            
            logger.info(f"✅ Position added for {symbol}")
        except Exception as e:
            logger.error(f"❌ Error adding position for {symbol}: {str(e)}")  

    def remove_position(self, symbol):
        if symbol in self.positions:
            del self.positions[symbol]