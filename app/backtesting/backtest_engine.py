import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import asyncio
from ..models.stock import StockAnalysis
from ..models.news import NewsArticle
from ..models.position import Position
from ..services.stock_analyzer import StockAnalyzer
from ..services.sentiment_analyzer import SentimentAnalyzer
import aiohttp
import logging
import os

logger = logging.getLogger(__name__)

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')  # Use demo key if not set

class BacktestResult:
    def __init__(self):
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_return = 0.0
        self.max_drawdown = 0.0
        self.positions: List[Position] = []
        self.trade_history: List[Dict] = []
        self.spy_return = 0.0  # S&P500 return for the same period

class BacktestEngine:
    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date = end_date
        self.stock_analyzer = StockAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.result = BacktestResult()
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    async def get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close(self):
        """Close the session"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def _fetch_historical_news(self, symbol: str, date: datetime) -> List[NewsArticle]:
        """Fetch historical news using Yahoo Finance"""
        try:
            # Use yfinance to get news
            stock = yf.Ticker(symbol)
            news_data = stock.news
            
            news_articles = []
            if news_data:
                print(f"Found {len(news_data)} news articles for {symbol}")
                for item in news_data:
                    try:
                        # Convert timestamp to datetime
                        news_date = datetime.fromtimestamp(item['providerPublishTime'], tz=timezone.utc)
                        
                        # Include news from a 3-day window around our target date
                        target_date = date.date()
                        news_window = timedelta(days=3)
                        if abs(news_date.date() - target_date) <= news_window.days:
                            article = NewsArticle(
                                title=item['title'],
                                content=item.get('summary', ''),
                                source=item.get('publisher', 'Yahoo Finance'),
                                url=item.get('link', ''),
                                published_at=news_date,
                                related_symbols=[symbol]
                            )
                            
                            # Calculate sentiment
                            article.sentiment_score = self.sentiment_analyzer.analyze_sentiment(article)
                            print(f"Article for {symbol}: {article.title[:50]}... | Sentiment: {article.sentiment_score:.2f}")
                            news_articles.append(article)
                    except Exception as e:
                        logger.error(f"Error processing news item: {str(e)}")
                        continue
            
            return news_articles
                    
        except Exception as e:
            logger.error(f"Error fetching news for {symbol} on {date}: {str(e)}")
            return []

    def _generate_simulated_news(self, symbol: str, hist_data: pd.DataFrame, date: datetime) -> List[NewsArticle]:
        """Generate simulated news based on price movements"""
        news_articles = []
        
        try:
            # Get the first row of data
            row = hist_data.iloc[0]
            
            # Calculate daily return
            daily_return = float((row['Close'] - row['Open']) / row['Open'])
            
            # Calculate volume ratio
            volume_ratio = float(row['Volume'] / hist_data['Volume'].mean())
            
            # Calculate price range percentage
            range_pct = float((row['High'] - row['Low']) / row['Open'])
            
            # Generate news based on significant price movements
            if abs(daily_return) > 0.005 or volume_ratio > 1.1 or range_pct > 0.02:
                sentiment = 1.0 if daily_return > 0 else -1.0
                impact = min(abs(daily_return) * 5, 1.0)
                
                title = f"{symbol} {'Surges' if daily_return > 0 else 'Declines'} on {'High' if volume_ratio > 1.5 else 'Normal'} Volume"
                content = f"Stock shows significant {'upward' if daily_return > 0 else 'downward'} movement with {volume_ratio:.1f}x normal volume."
                
                article = NewsArticle(
                    title=title,
                    content=content,
                    source="Market Data",
                    url="",
                    published_at=date,
                    related_symbols=[symbol]
                )
                article.sentiment_score = sentiment * impact
                news_articles.append(article)
                
                print(f"Generated news for {symbol}: {title} | Sentiment: {article.sentiment_score:.2f}")
        
        except Exception as e:
            logger.error(f"Error generating simulated news for {symbol}: {str(e)}")
        
        return news_articles

    async def run_backtest(self, symbols: List[str]) -> BacktestResult:
        """Run backtest for given symbols over the specified period"""
        try:
            # First, calculate S&P500 return for the period
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(start=self.start_date, end=self.end_date)
            if not spy_hist.empty:
                spy_start = spy_hist['Close'].iloc[0]
                spy_end = spy_hist['Close'].iloc[-1]
                self.result.spy_return = ((spy_end - spy_start) / spy_start) * 100
        except Exception as e:
            print(f"Error calculating S&P500 return: {str(e)}")
            self.result.spy_return = 0.0

        # Dictionary to track sentiment scores for all stocks
        stock_sentiments = {}
        
        # First pass: Calculate initial sentiment for all stocks
        print("\nAnalyzing market sentiment...")
        total_stocks = len(symbols)
        processed = 0
        
        for symbol in symbols:
            try:
                processed += 1
                if processed % 50 == 0:
                    print(f"Progress: {processed}/{total_stocks} stocks analyzed")
                
                # Skip any symbols that look invalid
                if not symbol or len(symbol) > 5:
                    continue
                
                stock = yf.Ticker(symbol)
                hist_data = stock.history(
                    start=self.start_date,
                    end=self.start_date + timedelta(days=5),  # Get first week's data
                    interval='1d'
                )
                
                if hist_data.empty or len(hist_data) < 3:  # Require at least 3 days of data
                    continue
                
                # Calculate sentiment based on price movements
                daily_returns = (hist_data['Close'] - hist_data['Open']) / hist_data['Open']
                volume_ratios = hist_data['Volume'] / hist_data['Volume'].mean()
                
                # Calculate average sentiment
                sentiment_scores = []
                for date, ret, vol in zip(hist_data.index, daily_returns, volume_ratios):
                    if abs(ret) > 0.005 or vol > 1.1:
                        sentiment = 1.0 if ret > 0 else -1.0
                        impact = min(abs(ret) * 5, 1.0)
                        sentiment_scores.append(sentiment * impact)
                
                if sentiment_scores:
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                # Generate simulated news based on price movements
                total_sentiment = 0
                sentiment_count = 0
                
                for date in hist_data.index:
                    daily_data = hist_data.loc[[date]]
                    news = self._generate_simulated_news(symbol, daily_data, date)
                    if news:
                        day_sentiment = sum(n.sentiment_score for n in news) / len(news)
                        total_sentiment += day_sentiment
                        sentiment_count += 1
                
                if sentiment_count > 0:
                    avg_sentiment = total_sentiment / sentiment_count
                    avg_volume = hist_data['Volume'].mean()
                    baseline_volume = hist_data['Volume'].iloc[0]  # Use first day's volume as baseline
                    avg_volume_ratio = avg_volume / baseline_volume if baseline_volume > 0 else 1.0
                    
                    stock_sentiments[symbol] = {
                        'sentiment': avg_sentiment,
                        'volume_ratio': avg_volume_ratio,
                        'price': hist_data['Close'].iloc[-1]
                    }
                    print(f"Added sentiment for {symbol}: {avg_sentiment:.2f}")
            
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                continue
        
        # Sort stocks by sentiment and volume ratio
        ranked_stocks = sorted(
            stock_sentiments.items(),
            key=lambda x: (x[1]['sentiment'], x[1]['volume_ratio']),
            reverse=True
        )
        
        # Select stocks with any positive sentiment
        selected_stocks = [
            symbol for symbol, data in ranked_stocks[:100]  # Look at more stocks
            if data['sentiment'] > 0.0  # Any positive sentiment
        ][:30]  # Take top 30
        
        print(f"\nSelected {len(selected_stocks)} stocks for trading based on sentiment analysis:")
        for symbol in selected_stocks:
            print(f"{symbol}: Initial Sentiment = {stock_sentiments[symbol]['sentiment']:.2f}, "
                  f"Volume Ratio = {stock_sentiments[symbol]['volume_ratio']:.2f}")
        
        if not selected_stocks:
            print("\nNo stocks met the sentiment criteria. Debug info:")
            print(f"Total stocks with sentiment data: {len(stock_sentiments)}")
            if stock_sentiments:
                print("\nTop 5 stocks by sentiment:")
                for symbol, data in sorted(stock_sentiments.items(), 
                                        key=lambda x: x[1]['sentiment'], 
                                        reverse=True)[:5]:
                    print(f"{symbol}: Sentiment = {data['sentiment']:.2f}")
        
        # Second pass: Detailed backtesting of selected stocks
        print("\nRunning detailed backtest on selected stocks...")
        for symbol in selected_stocks:
            try:
                stock = yf.Ticker(symbol)
                hist_data = stock.history(
                    start=self.start_date,
                    end=self.end_date,
                    interval='1d'
                )
                
                current_position = None
                
                for date, row in hist_data.iterrows():
                    # Generate simulated news based on daily price action
                    daily_data = hist_data.loc[[date]]
                    news_articles = self._generate_simulated_news(symbol, daily_data, date)
                    analysis = self._create_analysis(symbol, row, news_articles)
                    
                    if current_position is None:
                        if analysis.should_buy():
                            current_position = Position(
                                symbol=symbol,
                                entry_price=row['Close'],
                                entry_time=date.to_pydatetime(),
                                entry_sentiment=analysis.sentiment_score,
                                entry_confidence=analysis.confidence_score,
                                current_price=row['Close'],
                                current_sentiment=analysis.sentiment_score,
                                current_confidence=analysis.confidence_score,
                                last_updated=date.to_pydatetime()
                            )
                            self.result.trade_history.append({
                                'date': date,
                                'symbol': symbol,
                                'action': 'BUY',
                                'price': row['Close'],
                                'sentiment': analysis.sentiment_score
                            })
                    else:
                        current_position.current_price = row['Close']
                        current_position.current_sentiment = analysis.sentiment_score
                        current_position.current_confidence = analysis.confidence_score
                        current_position.last_updated = date.to_pydatetime()
                        
                        if analysis.should_sell(None):
                            profit_loss = current_position.profit_loss
                            self.result.total_return += profit_loss
                            
                            if profit_loss > 0:
                                self.result.winning_trades += 1
                            else:
                                self.result.losing_trades += 1
                            
                            self.result.total_trades += 1
                            self.result.positions.append(current_position)
                            
                            self.result.trade_history.append({
                                'date': date,
                                'symbol': symbol,
                                'action': 'SELL',
                                'price': row['Close'],
                                'sentiment': analysis.sentiment_score,
                                'profit_loss': profit_loss
                            })
                            
                            current_position = None
                
                if current_position is not None:
                    profit_loss = current_position.profit_loss
                    self.result.total_return += profit_loss
                    self.result.total_trades += 1
                    if profit_loss > 0:
                        self.result.winning_trades += 1
                    else:
                        self.result.losing_trades += 1
                    
                    self.result.positions.append(current_position)
                    self.result.trade_history.append({
                        'date': self.end_date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'price': hist_data.iloc[-1]['Close'],
                        'sentiment': 0,
                        'profit_loss': profit_loss
                    })
            
            except Exception as e:
                print(f"Error backtesting {symbol}: {str(e)}")
                continue
        
        # Calculate max drawdown
        self.result.max_drawdown = self._calculate_max_drawdown()
        
        # Clean up
        await self.close()
        
        return self.result

    def _create_analysis(self, symbol: str, data: pd.Series, news_articles: List[NewsArticle]) -> StockAnalysis:
        """Create a stock analysis object using historical data and news"""
        analysis = StockAnalysis(symbol, data['Close'], news_articles)
        
        try:
            # Calculate technical indicators using the historical data frame
            hist_data = data.to_frame().T  # Convert Series to DataFrame
            
            # Volume ratio (current volume vs average)
            volume_ma = hist_data['Volume'].mean()
            analysis.volume_ratio = hist_data['Volume'].iloc[0] / volume_ma if volume_ma != 0 else 1.0
            
            # Price momentum (compare to previous close)
            analysis.price_momentum = 0.0  # Default to neutral momentum
            
            # Set sentiment scores based on actual news
            if news_articles:
                # Weight recent news more heavily
                total_weight = 0
                weighted_sentiment = 0
                for article in news_articles:
                    age_weight = 1.0  # Historical news all get equal weight since we're backtesting
                    weighted_sentiment += article.sentiment_score * age_weight
                    total_weight += age_weight
                
                if total_weight > 0:
                    analysis.sentiment_score = weighted_sentiment / total_weight
                    # More aggressive impact and confidence scores
                    analysis.news_impact_score = min(abs(analysis.sentiment_score) * 3.0, 1.0)  # Increased from 2.5
                    analysis.confidence_score = min(abs(analysis.sentiment_score) * analysis.news_impact_score * 2.5, 1.0)  # Increased from 2.0
            
            # Set technical indicators to be more sensitive to sentiment
            analysis.rsi = 70.0 if analysis.sentiment_score > 0 else 30.0  # More extreme RSI values
            analysis.macd = analysis.sentiment_score * 1.2  # Increased from 0.8
            analysis.ma_50_200_ratio = 1.0 + (analysis.sentiment_score * 0.2)  # Increased from 0.15
            
        except Exception as e:
            print(f"Error calculating technical indicators for {symbol}: {str(e)}")
            # Return analysis with default values
            
        return analysis
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from trade history"""
        if not self.result.trade_history:
            return 0.0
        
        peak = float('-inf')
        max_drawdown = 0.0
        current_return = 0.0
        
        for trade in self.result.trade_history:
            if trade['action'] == 'SELL':
                current_return += trade['profit_loss']
                peak = max(peak, current_return)
                drawdown = peak - current_return
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown

    def print_results(self):
        """Print backtest results in a formatted way"""
        print("\n=== Backtest Results ===")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Total Trades: {self.result.total_trades}")
        if self.result.total_trades > 0:
            win_rate = (self.result.winning_trades / self.result.total_trades) * 100
            print(f"Win Rate: {win_rate:.1f}%")
        print(f"Strategy Return: {self.result.total_return:.2f}%")
        print(f"S&P500 Return: {self.result.spy_return:.2f}%")
        print(f"Excess Return: {(self.result.total_return - self.result.spy_return):.2f}%")
        print(f"Max Drawdown: {self.result.max_drawdown:.2f}%")
        
        print("\nTrade History:")
        print("=" * 80)
        for trade in self.result.trade_history:
            print(
                f"{trade['date'].date()} | {trade['symbol']} | {trade['action']} | "
                f"${trade['price']:.2f} | Sentiment: {trade['sentiment']:.2f}"
                + (f" | P/L: {trade['profit_loss']:.2f}%" if 'profit_loss' in trade else "")
            ) 