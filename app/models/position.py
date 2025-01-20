import pytz         # Also add this if not already there
from datetime import datetime, timezone, timedelta

class Position:
    def __init__(self, symbol, entry_price, target_price, target_date, timeframe):
        self.symbol = symbol
        self.entry_price = entry_price
        self.current_price = entry_price
        self.target_price = target_price
        self.target_date = target_date
        self.timeframe = timeframe
        self.entry_date = datetime.now(tz=pytz.UTC)
        return self.current_sentiment - self.entry_sentiment 