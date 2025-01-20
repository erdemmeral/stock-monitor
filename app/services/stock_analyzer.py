from datetime import datetime, timezone
from typing import List, Optional
import yfinance as yf
from ..models.news import NewsArticle
from ..models.stock import StockAnalysis
import logging
import math
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class StockAnalyzer:
    def __init__(self):
        self.market_etf = "SPY"  # S&P 500 ETF for market sentiment
        self.sector_etfs = {
            "XLK": ["AAPL", "MSFT", "NVDA"],  # Technology
            "XLF": ["JPM", "BAC", "WFC"],     # Financials
            "XLE": ["XOM", "CVX", "COP"],     # Energy
            "XLV": ["JNJ", "UNH", "PFE"],     # Healthcare
            "XLY": ["AMZN", "TSLA", "HD"],    # Consumer Discretionary
            # Add more sectors as needed
        }
        self.cached_market_data = {}
        self.cache_expiry = 300  # 5 minutes

    def analyze_stock(self, symbol: str, news_articles: List[NewsArticle]) -> Optional[StockAnalysis]:
        """Analyze a stock based on news and technical indicators"""
        try:
            # Get stock data
            stock = yf.Ticker(symbol)
            
            # Get historical data for technical analysis
            hist = stock.history(period="200d")
            if hist.empty:
                return None
                
            current_price = hist['Close'].iloc[-1]
            
            # Create analysis object
            analysis = StockAnalysis(symbol, current_price, news_articles)
            
            # Calculate technical indicators
            self._calculate_technical_indicators(analysis, hist)
            
            # Calculate sentiment scores
            self._calculate_sentiment_scores(analysis, news_articles)
            
            # Calculate market conditions
            self._calculate_market_conditions(analysis, symbol)
            
            # Calculate risk metrics
            self._calculate_risk_metrics(analysis, hist)
            
            # Update final recommendation
            analysis.update_recommendation()
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing stock {symbol}: {str(e)}")
            return None

    def _calculate_technical_indicators(self, analysis: StockAnalysis, hist: pd.DataFrame):
        """Calculate technical indicators"""
        try:
            # Calculate RSI (14-day)
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            analysis.rsi = 100 - (100 / (1 + rs.iloc[-1]))
            
            # Calculate MACD
            exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
            exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            analysis.macd = macd.iloc[-1] - signal.iloc[-1]
            
            # Calculate Moving Averages
            ma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
            ma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
            analysis.ma_50_200_ratio = ma_50 / ma_200 if ma_200 != 0 else 1.0
            
            # Calculate volume ratio
            avg_volume = hist['Volume'].rolling(window=20).mean().iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            analysis.volume_ratio = current_volume / avg_volume if avg_volume != 0 else 1.0
            
            # Calculate price momentum (20-day return)
            analysis.price_momentum = (
                hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1
                if len(hist) >= 20 else 0.0
            )
            
            # Calculate volatility (20-day standard deviation of returns)
            returns = hist['Close'].pct_change()
            analysis.volatility = returns.rolling(window=20).std().iloc[-1]
            
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")

    def _calculate_sentiment_scores(self, analysis: StockAnalysis, news_articles: List[NewsArticle]):
        """Calculate sentiment scores from news articles"""
        if not news_articles:
            return
            
        # Calculate average sentiment
        sentiments = [article.sentiment_score for article in news_articles if article.sentiment_score is not None]
        analysis.sentiment_score = np.mean(sentiments) if sentiments else 0.0
        
        # Calculate news impact score based on source reliability and recency
        total_impact = 0
        total_weight = 0
        
        for article in news_articles:
            if article.sentiment_score is not None:
                # Calculate time decay (more recent = higher weight)
                hours_old = (datetime.now(article.published_at.tzinfo) - article.published_at).total_seconds() / 3600
                time_weight = 1.0 / (1.0 + hours_old/24.0)  # Decay over 24 hours
                
                total_impact += abs(article.sentiment_score) * time_weight
                total_weight += time_weight
        
        analysis.news_impact_score = total_impact / total_weight if total_weight > 0 else 0.0
        
        # Calculate confidence score based on consensus
        if sentiments:
            sentiment_std = np.std(sentiments)
            analysis.confidence_score = 1.0 - min(sentiment_std, 1.0)  # Higher consensus = higher confidence
        else:
            analysis.confidence_score = 0.0

    def _calculate_market_conditions(self, analysis: StockAnalysis, symbol: str):
        """Calculate market and sector sentiment with focus on recent data"""
        try:
            # Get market sentiment (S&P 500)
            market_data = self._get_cached_data(self.market_etf)
            if market_data is not None and not market_data.empty:
                # Calculate returns
                market_returns = market_data['Close'].pct_change()
                if not market_returns.empty:
                    # Weight recent returns more heavily
                    if len(market_returns) >= 2:
                        # Most recent day gets 70% weight, previous day gets 30%
                        analysis.market_sentiment = (
                            market_returns.iloc[-1] * 0.7 + 
                            market_returns.iloc[-2] * 0.3
                        ) * 100
                    else:
                        analysis.market_sentiment = market_returns.iloc[-1] * 100
                else:
                    analysis.market_sentiment = 0.0
            else:
                analysis.market_sentiment = 0.0
            
            # Get sector sentiment
            sector_etf = None
            for etf, symbols in self.sector_etfs.items():
                if symbol in symbols:
                    sector_etf = etf
                    break
            
            if sector_etf:
                sector_data = self._get_cached_data(sector_etf)
                if sector_data is not None and not sector_data.empty:
                    sector_returns = sector_data['Close'].pct_change()
                    if not sector_returns.empty:
                        # Weight recent returns more heavily
                        if len(sector_returns) >= 2:
                            # Most recent day gets 70% weight, previous day gets 30%
                            analysis.sector_sentiment = (
                                sector_returns.iloc[-1] * 0.7 + 
                                sector_returns.iloc[-2] * 0.3
                            ) * 100
                        else:
                            analysis.sector_sentiment = sector_returns.iloc[-1] * 100
                    else:
                        analysis.sector_sentiment = 0.0
                else:
                    analysis.sector_sentiment = 0.0
            else:
                analysis.sector_sentiment = 0.0
            
        except Exception as e:
            logger.error(f"Error calculating market conditions: {str(e)}")
            analysis.market_sentiment = 0.0
            analysis.sector_sentiment = 0.0

    def _calculate_risk_metrics(self, analysis: StockAnalysis, hist: pd.DataFrame):
        """Calculate risk metrics"""
        try:
            # Calculate maximum drawdown
            rolling_max = hist['Close'].rolling(window=252, min_periods=1).max()
            daily_drawdown = hist['Close'] / rolling_max - 1.0
            analysis.max_drawdown = daily_drawdown.min()
            
            # Calculate composite risk score based on multiple factors
            volatility_risk = min(analysis.volatility * 100, 1.0)  # Scale volatility to 0-1
            drawdown_risk = min(abs(analysis.max_drawdown), 1.0)
            momentum_risk = min(abs(analysis.price_momentum), 1.0)
            
            # Combine risk factors (weighted average)
            analysis.risk_score = (
                volatility_risk * 0.4 +
                drawdown_risk * 0.4 +
                momentum_risk * 0.2
            )
            
        except Exception as e:
            print(f"Error calculating risk metrics: {str(e)}")

    def _get_cached_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get cached market data or fetch new data"""
        now = datetime.now()
        
        # Check if we have fresh cached data
        if symbol in self.cached_market_data:
            cache_time, data = self.cached_market_data[symbol]
            if (now - cache_time).total_seconds() < self.cache_expiry:
                return data
        
        # Fetch new data
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="5d")
            self.cached_market_data[symbol] = (now, data)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None 