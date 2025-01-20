from datetime import datetime, timezone, timedelta
from typing import List, Optional
from .news import NewsArticle

class StockAnalysis:
    def __init__(self, symbol: str, current_price: float, news_articles: List[NewsArticle]):
        self.symbol = symbol
        self.current_price = current_price
        self.news_articles = news_articles
        self.sentiment_score = 0.0
        self.news_impact_score = 0.0
        self.confidence_score = 0.0
        self.recommendation = "neutral"
        self.analysis_timestamp = datetime.now()
        self.volume_ratio = 1.0  # Current volume compared to average
        self.price_momentum = 0.0  # Recent price momentum
        self.volatility = 0.0  # Recent price volatility
        
        # Technical indicators
        self.rsi = 50.0  # Relative Strength Index
        self.macd = 0.0  # Moving Average Convergence Divergence
        self.ma_50_200_ratio = 1.0  # 50-day MA / 200-day MA
        
        # Market conditions
        self.market_sentiment = 0.0  # Overall market sentiment
        self.sector_sentiment = 0.0  # Sector-specific sentiment
        
        # Risk metrics
        self.risk_score = 0.0  # Composite risk score
        self.max_drawdown = 0.0  # Maximum historical drawdown

    def should_buy(self) -> bool:
        """
        Determine if a buy signal should be generated based on multiple factors
        """
        # Get current market hour (ET)
        current_time = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-4)))
        
        # Determine market session
        is_premarket = (
            current_time.hour < 9 or 
            (current_time.hour == 9 and current_time.minute < 30)
        )
        is_afterhours = (
            current_time.hour > 16 or
            (current_time.hour == 16 and current_time.minute > 0)
        )
        
        # Core sentiment requirements (lowered by ~5%)
        sentiment_check = (
            self.sentiment_score > 0.28 and     # Was 0.3
            self.news_impact_score > 0.47 and   # Was 0.5
            self.confidence_score > 0.38        # Was 0.4
        )
        
        # Technical analysis requirements (lowered by ~5%)
        technical_check = (
            self.rsi < 73 and                   # Was 70
            self.macd > -0.05 and              # Was 0
            self.ma_50_200_ratio > 0.90        # Was 0.95
        )
        
        # Volume and momentum requirements - adjusted for market session
        if is_premarket:
            volume_threshold = 0.5      # 50% of average volume for pre-market
        elif is_afterhours:
            volume_threshold = 0.6      # 60% of average volume for after-hours
        else:
            volume_threshold = 1.14     # 114% of average volume for regular hours
            
        volume_check = (
            self.volume_ratio > volume_threshold and
            self.price_momentum > -0.02        # Was 0
        )
        
        # Risk assessment (increased tolerance by ~5%)
        risk_check = (
            self.risk_score < 0.74 and         # Was 0.7
            self.volatility < 0.32             # Was 0.3
        )
        
        # Market conditions (slightly more tolerant)
        market_check = (
            self.market_sentiment >= -0.05 and  # Was 0
            self.sector_sentiment >= -0.05      # Was 0
        )
        
        # Combined decision with weights
        core_signal = sentiment_check and technical_check
        supporting_signals = sum([volume_check, risk_check, market_check])
        
        return core_signal and supporting_signals >= 2

    def should_sell(self, prev_analysis: Optional['StockAnalysis'] = None) -> tuple[bool, list[str]]:
        """
        Determine if a sell signal should be generated.
        Returns (should_sell, list of reasons)
        """
        if not prev_analysis:
            return False, []
            
        sell_reasons = []
        
        # Sentiment deterioration checks
        if self.sentiment_score < -0.2:
            sell_reasons.append(f"Negative sentiment score ({self.sentiment_score:.2f})")
        elif (self.sentiment_score - prev_analysis.sentiment_score) < -0.3:
            sell_reasons.append(
                f"Sharp sentiment decline ({prev_analysis.sentiment_score:.2f} → {self.sentiment_score:.2f})"
            )
        
        # Technical breakdown checks
        if self.rsi > 70:
            sell_reasons.append(f"Overbought RSI ({self.rsi:.1f})")
        if self.macd < 0:
            sell_reasons.append(f"Negative MACD ({self.macd:.3f})")
        if self.ma_50_200_ratio < 0.95:
            sell_reasons.append(f"Bearish MA crossover (ratio: {self.ma_50_200_ratio:.2f})")
        
        # Volume warning checks
        if self.volume_ratio > 2.0 and self.price_momentum < 0:
            sell_reasons.append(
                f"High volume ({self.volume_ratio:.1f}x) with negative momentum ({self.price_momentum:.2f})"
            )
        
        # Risk elevation checks
        if self.risk_score > 0.8:
            sell_reasons.append(f"High risk score ({self.risk_score:.2f})")
        if self.volatility > 0.4:
            sell_reasons.append(f"High volatility ({self.volatility:.2f})")
        
        # Market condition checks
        if self.market_sentiment < -0.2:
            sell_reasons.append(f"Negative market sentiment ({self.market_sentiment:.2f})")
        if self.sector_sentiment < -0.2:
            sell_reasons.append(f"Negative sector sentiment ({self.sector_sentiment:.2f})")
        
        # Determine if we should sell based on the reasons
        should_sell = False
        
        # Primary signals (sentiment or technical)
        has_primary = any(
            "sentiment" in reason.lower() or 
            any(tech in reason.lower() for tech in ["rsi", "macd", "ma crossover"])
            for reason in sell_reasons
        )
        
        # Supporting signals (volume, risk, market)
        supporting_signals = sum(
            1 for reason in sell_reasons
            if any(signal in reason.lower() for signal in ["volume", "risk", "volatility", "market", "sector"])
        )
        
        should_sell = has_primary and supporting_signals >= 1
        
        return should_sell, sell_reasons

    def update_recommendation(self):
        """Update the stock recommendation based on analysis"""
        if self.sentiment_score >= 0.5 and self.confidence_score >= 0.6:
            self.recommendation = "strong_buy"
        elif self.sentiment_score >= 0.3 and self.confidence_score >= 0.4:
            self.recommendation = "buy"
        elif self.sentiment_score <= -0.5 and self.confidence_score >= 0.6:
            self.recommendation = "strong_sell"
        elif self.sentiment_score <= -0.3 and self.confidence_score >= 0.4:
            self.recommendation = "sell"
        else:
            self.recommendation = "neutral" 