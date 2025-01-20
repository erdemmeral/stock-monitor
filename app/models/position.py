from datetime import datetime
import pytz

class Position:
    def __init__(self, symbol, entry_price, current_price, target_price, entry_date, target_date, timeframe):
        self.symbol = symbol
        self.entry_price = entry_price
        self.current_price = current_price
        self.target_price = target_price
        self.entry_date = entry_date  # This should be a timezone-aware datetime
        self.target_date = target_date
        self.timeframe = timeframe

    def get_position_duration(self):
        """Calculate the duration of the position"""
        current_time = datetime.now(tz=pytz.UTC)
        duration = current_time - self.entry_date
        
        # Convert to days, hours, and minutes for readability
        days = duration.days
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        return {
            'days': days,
            'hours': hours,
            'minutes': minutes,
            'total_seconds': duration.total_seconds()
        }