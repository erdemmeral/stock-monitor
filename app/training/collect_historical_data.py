import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import pytz

def collect_training_data(symbols, lookback_days=365):
    training_data = []
    
    for symbol in symbols:
        stock = yf.Ticker(symbol)
        news = stock.news
        prices = stock.history(period=f"{lookback_days}d", interval="1h")
        
        for article in news:
            # Convert timestamp to timezone-aware datetime
            pub_date = datetime.fromtimestamp(article['providerPublishTime'], tz=pytz.UTC)
            
            # Get next day's price movement
            try:
                start_price = prices['Close'].loc[pub_date]
                end_price = prices['Close'].loc[pub_date + timedelta(days=1)]
                price_impact = (end_price - start_price) / start_price
                
                training_data.append({
                    'news_text': article['title'],
                    'market_impact': 1 if price_impact > 0.02 else (0 if price_impact < -0.02 else 2),
                    'price_change': price_impact,
                    'symbol': symbol
                })
            except KeyError:
                continue

    return pd.DataFrame(training_data)

# Train on major stocks
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'PFE', 'JNJ', 'UNH']
df = collect_training_data(symbols)
# Save to the correct path
df.to_csv('app/training/historical_news.csv', index=False)
