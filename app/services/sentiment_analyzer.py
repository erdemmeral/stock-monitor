from bs4 import BeautifulSoup
import requests
import joblib
import logging
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timezone
import re
from typing import Optional
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sentiment_analyzer.log')
    ]
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        logger.info("Loading ML models...")
        try:
            # Load vectorizer
            self.vectorizer = joblib.load('app/models/vectorizer.joblib')
            logger.info("✨ Vectorizer loaded successfully!")
            
            # Load timeframe-specific models
            self.models = {
                '1h': joblib.load('app/models/market_model_1h.joblib'),
                '1w': joblib.load('app/models/market_model_1wk.joblib'),
                '1m': joblib.load('app/models/market_model_1mo.joblib')
            }
            logger.info("✨ All timeframe models loaded successfully!")
            
            self.thresholds = {
                '1h': 5.0,
                '1w': 10.0,
                '1m': 20.0
            }
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def analyze_sentiment(self, article):
        try:
            # Combine title and summary for analysis
            text = f"{article['title']} {article.get('summary', '')}"
            X = self.vectorizer.transform([text])
            
            # Get predictions for each timeframe
            predictions = {}
            for timeframe, model in self.models.items():
                pred = model.predict(X)[0]
                predictions[timeframe] = pred
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return None
    def _get_base_sentiment(self, article) -> float:
        if not article.title and not article.content:
            return 0.0
        
        # Combine title and content
        text = f"{article.title}. {article.content}"
        
        # Clean and preprocess text
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = text.lower()
        
        # Use TextBlob for initial sentiment
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        
        # Use VADER for more advanced sentiment analysis
        vs = self.vader.polarity_scores(text)
        vader_sentiment = vs['compound']
        
        # Weight VADER sentiment higher
        weighted_sentiment = (sentiment + 2 * vader_sentiment) / 3
        
        # Adjust sentiment based on source reliability
        source_weight = 1.0
        if article.source:
            source = article.source.lower()
            if any(s in source for s in ['prnewswire', 'businesswire', 'reuters', 'bloomberg', 'wsj', 'ft.com']):
                source_weight = 1.2  # Increase weight for reputable sources
            elif any(s in source for s in ['seekingalpha', 'zacks', 'fool.com', 'barrons', 'investors.com']):
                source_weight = 0.8  # Decrease weight for less reliable sources
        
        adjusted_sentiment = weighted_sentiment * source_weight
        
        # Check for key phrases that indicate important positive news
        title_lower = article.title.lower()
        content_lower = article.content.lower() if article.content else ""
        
        if any(phrase in title_lower or phrase in content_lower for phrase in [
            'new distribution center',
            'approvals',
            'acquisition',
            'completes acquisition',
            'investment',
            'contract win',
            'wins contract',
            'new product',
            'launches new',
            'expanding',
            'expansion'
        ]):
            adjusted_sentiment = min(adjusted_sentiment * 1.5, 1.0)  # Boost sentiment for key phrases
        
        return adjusted_sentiment