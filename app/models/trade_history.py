from datetime import datetime
import pytz
import json
import os

class TradeHistory:
    def __init__(self, file_path='trading_history.json'):
        self.file_path = file_path
        self.trades = []
        self.load_history()

    def add_trade(self, symbol, entry_price, exit_price, entry_date, exit_date, 
                 timeframe, target_price, reason, profit_loss):
        trade = {
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_date': entry_date.isoformat(),
            'exit_date': exit_date.isoformat(),
            'timeframe': timeframe,
            'target_price': target_price,
            'reason': reason,
            'profit_loss': profit_loss,
            'profit_loss_percentage': ((exit_price - entry_price) / entry_price) * 100
        }
        self.trades.append(trade)
        self.save_history()
        return trade

    def get_total_profit_loss(self):
        return sum(trade['profit_loss'] for trade in self.trades)

    def get_win_rate(self):
        if not self.trades:
            return 0
        winning_trades = sum(1 for trade in self.trades if trade['profit_loss'] > 0)
        return (winning_trades / len(self.trades)) * 100

    def get_trades_by_symbol(self, symbol):
        return [trade for trade in self.trades if trade['symbol'] == symbol]

    def save_history(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.trades, f, indent=4)

    def load_history(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                self.trades = json.load(f)