from app.models.position import Position
class Portfolio:
    def __init__(self):
        self.positions = {}  # symbol -> Position

    def add_position(self, symbol, entry_price, target_price, target_date, timeframe):
        self.positions[symbol] = Position(symbol, entry_price, target_price, target_date, timeframe)

    def remove_position(self, symbol):
        if symbol in self.positions:
            del self.positions[symbol]