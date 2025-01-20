from typing import List, Dict, Any
from datetime import datetime, timezone
from app.models.news_article import NewsArticle
from app.services.sentiment_analyzer import SentimentAnalyzer
from sklearn.feature_extraction.text import HashingVectorizer  # Memory efficient
from sklearn.linear_model import SGDClassifier  # Low memory footprint

class StockAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        # Use HashingVectorizer instead of TfidfVectorizer to save memory
        self.vectorizer = HashingVectorizer(n_features=2**16)
        # SGD classifier for online learning with minimal memory
        self.classifier = SGDClassifier(loss='log_loss')
        
        self.HIGH_PRIORITY_KEYWORDS = {
            'fda approval', 'merger complete', 'acquisition complete',
            'earnings beat', 'guidance raised', 'buyout approved',
            'major contract', 'strategic partnership',
            'breakthrough patent', 'clinical trial success',
            'sec investigation', 'ceo change',
            'major lawsuit', 'bankruptcy filing',
            'stock split announced'
        }
        
        self.MEDIUM_PRIORITY_KEYWORDS = {
            'partnership', 'collaboration agreement',
            'joint venture', 'clinical trial updates',
            'product launch', 'market expansion',
            'quarterly results', 'analyst upgrade',
            'new facility', 'executive appointment'
        }
        
        self.LOW_PRIORITY_KEYWORDS = {
            'webinar', 'conference presentation',
            'industry event', 'award received',
            'charity', 'sponsorship',
            'minor contract', 'routine update'
        }

        # Add new filter for speculative headlines
        self.SPECULATIVE_PHRASES = {
            'best stock', 'should you buy', 
            'could be the next', 'stocks to watch',
            'why you should', 'best performing stock',
            'stocks under', 'penny stocks',
            'stocks to buy', 'stocks that could',
            'top picks', 'bull run',
            'next big thing', 'potential multibagger'
        }

    def process_news(self, article: NewsArticle):
        text_features = self.vectorizer.transform([article.content])
        importance = self.classifier.predict_proba(text_features)[0]
        return importance * self.sentiment_analyzer.analyze_sentiment(article)
    def evaluate_news_importance(self, article: NewsArticle, sentiment_score: float) -> bool:
        title_lower = article.title.lower()
        content_lower = article.content.lower() if article.content else ""
        
        # Filter out speculative content first
        if any(phrase in title_lower for phrase in self.SPECULATIVE_PHRASES):
            return False
            
        # Continue with existing priority checks
        if any(keyword in title_lower or keyword in content_lower 
               for keyword in self.HIGH_PRIORITY_KEYWORDS):
            return True
            
        if any(keyword in title_lower or keyword in content_lower 
               for keyword in self.MEDIUM_PRIORITY_KEYWORDS):
            return abs(sentiment_score) > 0.6
            
        if any(keyword in title_lower or keyword in content_lower 
               for keyword in self.LOW_PRIORITY_KEYWORDS):
            return False
            
        return abs(sentiment_score) > 0.8

    def _calculate_sentiment_score(self, news_articles: List[NewsArticle]) -> float:
        logging.info("Starting ML-based sentiment analysis")
        for article in news_articles:
            score = self.sentiment_analyzer.analyze_sentiment(article)
            logging.info(f"Article: {article.title} | ML Score: {score}")
    
        if not news_articles:
            return 0.0
    
        sentiments = [
            self.sentiment_analyzer.analyze_sentiment(article)
            for article in news_articles
        ]

        now = datetime.now(timezone.utc)
        recent_articles = []
        old_articles = []

        # Separate and weight articles by time
        for i, article in enumerate(news_articles):
            hours_old = (now - article.published_at).total_seconds() / 3600
            sentiment_score = sentiments[i]
    
            if hours_old <= 24:
                # Recent news gets full weight
                recent_articles.append((sentiment_score, article))
            else:
                # Older news gets reduced weight based on age
                time_weight = 0.5 * (48 - min(hours_old, 48)) / 24
                weighted_score = sentiment_score * time_weight
                old_articles.append((weighted_score, article))

        # Check for any highly significant recent news
        for score, article in recent_articles:
            if score > 0.6:  # Lowered threshold for significant news
                return score * 1.2  # Return immediately with a boost
        
        # If no significant single news, proceed with weighted average
        if not recent_articles and not old_articles:
            return 0.0  # No sentiment if no articles
    
        # Calculate weighted scores
        weighted_sum = 0
        weight_sum = 0

        # Process recent articles with higher weights
        for i, (score, _) in enumerate(recent_articles):
            weight = 1.0 / (i + 1)  # Higher weight for more recent articles
            weighted_sum += score * weight
            weight_sum += weight

        # Process old articles with lower weights
        for i, (score, _) in enumerate(old_articles):
            weight = 0.3 / (i + 1)  # Lower weight for older articles
            weighted_sum += score * weight
            weight_sum += weight
    
        # Calculate final score
        if weight_sum > 0:
            final_score = weighted_sum / weight_sum
    
            # Apply multiplier based on number of recent positive articles
            positive_recent = sum(1 for score, _ in recent_articles if score > 0.5)  # Lowered threshold
            if positive_recent >= 2:
                final_score *= 1.2  # Increased boost for multiple good articles
    
            return final_score

        return 0.0  # No sentiment if no valid scores
def adjust_sentiment_score(self, sentiment_score: float, article: NewsArticle) -> float:
    title_lower = article.title.lower()
    content_lower = article.content.lower() if article.content else ""
    
    # Charity news sentiment reducers
    CHARITY_KEYWORDS = {
        'charitable foundation', 'grant cycle', 
        'community support', 'donation', 
        'philanthropy', 'nonprofit'
    }
    
    if any(keyword in title_lower or keyword in content_lower 
           for keyword in CHARITY_KEYWORDS):
        return sentiment_score * 0.2  # Will reduce 0.60 to 0.12
        
    return sentiment_score