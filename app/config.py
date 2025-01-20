import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")

# App Configuration
NEWS_UPDATE_INTERVAL = 300  # 5 minutes
MAX_NEWS_AGE_DAYS = 2
MIN_CONFIDENCE_SCORE = 0.6

# News Sources Configuration
TRUSTED_SOURCES = [
    "Reuters",
    "Bloomberg",
    "CNBC",
    "Financial Times",
    "Wall Street Journal"
]

# Stock Analysis Configuration
SENTIMENT_THRESHOLD = 0.6
IMPACT_SCORE_THRESHOLD = 0.7 