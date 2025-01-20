from requests_cache import logger
from app.models.position import Position
class Portfolio:
    def __init__(self):
        self.positions = {}  # symbol -> Position

    def add_position(self, symbol, entry_price, target_price, target_date, timeframe):

        logger.info(f"Attempting to add position for {symbol}")
        logger.info(f"Entry Price: ${entry_price:.2f}")
        logger.info(f"Target Price: ${target_price:.2f}")
        self.positions[symbol] = Position(symbol, entry_price, target_price, target_date, timeframe)

    def remove_position(self, symbol):
        if symbol in self.positions:
            del self.positions[symbol]